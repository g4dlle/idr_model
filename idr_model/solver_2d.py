"""
solver_2d.py -- iterative solver for the 2D axisymmetric IDR model.

Algorithm: same Gauss-Seidel over equations as 1D, but each equation
is a sparse linear system solved via scipy.sparse.linalg.spsolve.

Iteration order:
  1. Solve for |H|²  (sigma known from previous iteration)
  2. Compute |E_φ|² from Faraday's law per z-slice
  3. Update E_eff, Da, nu_i
  4. Solve for sigma profile (power iteration + normalisation)
  5. Under-relaxation + convergence check
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from equations_2d import make_grid_2d, build_H_equation_2d, build_sigma_equation_2d
from physics import (
    conductivity, effective_field,
    ambipolar_diffusion, ionization_freq,
)
from config import (
    N_GRID, R_TUBE, H_WALL, P_PA, N_E0,
    MAX_ITER, TOL, RELAX,
    E_CHARGE, M_ELECTRON, OMEGA,
)

MU_0 = 4.0 * np.pi * 1e-7


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ne_from_sigma_2d(sigma_a, p_pa):
    """Extract electron density from active conductivity."""
    from physics import collision_freq
    nu_c = collision_freq(p_pa)
    return sigma_a * M_ELECTRON * (nu_c**2 + OMEGA**2) / (E_CHARGE**2 * nu_c)


def compute_alpha_2d(sigma_a, sigma_p):
    """alpha = sigma_a / |sigma|^2"""
    mod2 = sigma_a**2 + sigma_p**2
    safe = np.where(mod2 > 0, mod2, 1e-300)
    return sigma_a / safe


def compute_E_faraday_2d(u_2d, r, hz, Nr, Nz):
    """
    Compute |E_φ|² from Faraday's law for each z-slice independently.

    E_φ(r) = ω·μ₀/r · ∫_0^r H(r')·r' dr'

    Parameters
    ----------
    u_2d : |H|², shape (Nr+1, Nz+1)
    r    : radial grid (Nr+1,)

    Returns
    -------
    v_2d : |E|², shape (Nr+1, Nz+1)
    """
    hr = r[1] - r[0] if len(r) > 1 else 1.0
    v_2d = np.zeros_like(u_2d)

    H_abs = np.sqrt(np.maximum(u_2d, 0.0))  # (Nr+1, Nz+1)

    for j in range(Nz + 1):
        integrand = H_abs[:, j] * r  # H(r') * r'
        intervals = 0.5 * hr * (integrand[:-1] + integrand[1:])
        integrals = np.zeros(Nr + 1)
        integrals[1:] = np.cumsum(intervals)
        # v[0, j] = 0 (E_φ = 0 on axis / conductor)
        v_2d[1:, j] = (OMEGA * MU_0 * integrals[1:] / r[1:])**2

    return v_2d


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_idr_2d(Nr=N_GRID, Nz=50, R=R_TUBE, L=0.05,
                 p_pa=P_PA, H_wall=H_WALL, n_e0=N_E0,
                 max_iter=MAX_ITER, tol=TOL, relax=RELAX,
                 r_inc=0.0, bc_z_sigma="dirichlet",
                 verbose=False):
    """
    Iterative solver for the 2D axisymmetric IDR model.

    Parameters
    ----------
    Nr         : grid intervals in r
    Nz         : grid intervals in z
    R          : tube radius [m]
    L          : tube length [m]
    p_pa       : pressure [Pa]
    H_wall     : H-field amplitude at the wall [A/m]
    n_e0       : initial electron density [m^-3]
    max_iter   : maximum iterations
    tol        : convergence criterion
    relax      : under-relaxation factor
    r_inc      : inner inclusion radius [m] (0 = no inclusion)
    bc_z_sigma : "dirichlet" or "neumann" for sigma at z-boundaries
    verbose    : print residuals

    Returns
    -------
    dict with keys: r, z, u, v, sigma_a, sigma_p, n_e, Da, nu_i,
                    converged, n_iter, residuals
    """
    r, z, hr, hz = make_grid_2d(Nr, Nz, R, L, r_min=r_inc)
    annular = r_inc > 0
    Nz1 = Nz + 1

    # ── Initialisation ────────────────────────────────────────────────────
    sigma_a0, sigma_p0, _ = conductivity(n_e0, p_pa)

    # Radial profile
    if annular:
        r_profile = (r - r_inc) * (R - r) / ((R - r_inc) / 2.0)**2
        r_profile = np.maximum(r_profile, 0.0)
    else:
        r_profile = np.maximum(1.0 - (r / R)**2, 0.0)

    # z-profile: parabolic z*(L-z) normalised
    if bc_z_sigma == "dirichlet":
        z_profile = z * (L - z) / (L / 2.0)**2
        z_profile = np.maximum(z_profile, 0.0)
    else:
        z_profile = np.ones_like(z)

    # 2D profile = outer product
    profile_2d = np.outer(r_profile, z_profile)  # (Nr+1, Nz+1)

    sigma_a = sigma_a0 * profile_2d
    sigma_p = sigma_p0 * profile_2d
    # Enforce BCs
    sigma_a[-1, :] = 0.0
    sigma_p[-1, :] = 0.0
    if annular:
        sigma_a[0, :] = 0.0
        sigma_p[0, :] = 0.0
    if bc_z_sigma == "dirichlet":
        sigma_a[:, 0] = 0.0
        sigma_a[:, -1] = 0.0
        sigma_p[:, 0] = 0.0
        sigma_p[:, -1] = 0.0

    u = np.zeros((Nr + 1, Nz1))
    v = np.zeros((Nr + 1, Nz1))
    u[-1, :] = H_wall**2

    residuals = []
    converged = False

    for iteration in range(max_iter):

        # Step a: solve for |H|²
        alpha = compute_alpha_2d(sigma_a, sigma_p)
        if np.all(alpha == 0):
            alpha = np.full((Nr + 1, Nz1), 1e-10)

        A_H, rhs_H = build_H_equation_2d(
            r, z, hr, hz, alpha, sigma_a, v, H_wall**2, bc_z="neumann"
        )
        u_flat = spsolve(A_H, rhs_H)
        u_new = np.maximum(u_flat.reshape(Nr + 1, Nz1), 0.0)

        # Step b: |E_φ|² from Faraday per z-slice
        v_new = compute_E_faraday_2d(u_new, r, hz, Nr, Nz)

        # Step c: effective field & transport coefficients
        E_abs = np.sqrt(v_new)
        E_eff = effective_field(E_abs, p_pa)
        Da = ambipolar_diffusion(E_eff, p_pa)
        nu_i = ionization_freq(E_eff, p_pa)

        # Step d: solve for sigma (power iteration + normalisation)
        A_s, rhs_s = build_sigma_equation_2d(
            r, z, hr, hz, Da, nu_i,
            sigma_ref=sigma_a, dt=None,
            bc_z_sigma=bc_z_sigma,
        )
        sigma_raw_flat = spsolve(A_s, rhs_s)
        sigma_raw = np.maximum(sigma_raw_flat.reshape(Nr + 1, Nz1), 0.0)

        # Enforce BCs
        sigma_raw[-1, :] = 0.0
        if annular:
            sigma_raw[0, :] = 0.0
        if bc_z_sigma == "dirichlet":
            sigma_raw[:, 0] = 0.0
            sigma_raw[:, -1] = 0.0

        # Normalisation: preserve integral ∫ n_e · r dr dz
        n_e_current = ne_from_sigma_2d(sigma_a, p_pa)

        # Radial weight for cylindrical integral
        r_col = r[:, np.newaxis]  # (Nr+1, 1) for broadcasting
        integral_ref = np.trapezoid(np.trapezoid(n_e_current * r_col, z, axis=1), r)
        target = integral_ref if integral_ref > 1e-300 else n_e0 * R**2 * L / 4.0

        integral_new = np.trapezoid(np.trapezoid(sigma_raw * r_col, z, axis=1), r)
        norm = integral_new if integral_new > 1e-300 else 1.0

        n_e_new = sigma_raw * target / norm
        n_e_new[-1, :] = 0.0
        if annular:
            n_e_new[0, :] = 0.0
        if bc_z_sigma == "dirichlet":
            n_e_new[:, 0] = 0.0
            n_e_new[:, -1] = 0.0

        sigma_a_new, sigma_p_new, _ = conductivity(n_e_new, p_pa)

        # Under-relaxation
        sigma_a_next = relax * sigma_a_new + (1.0 - relax) * sigma_a
        sigma_p_next = relax * sigma_p_new + (1.0 - relax) * sigma_p

        # Convergence check: shape + field
        if np.any(u[:, :] > 0):
            safe_u = np.where(np.abs(u) > 1e-30, np.abs(u), 1e-30)
            res_u = float(np.min([np.max(np.abs(u_new - u) / safe_u), 1e20]))
        else:
            res_u = 1.0

        s_max_new = np.max(sigma_a_new) if np.max(sigma_a_new) > 0 else 1.0
        s_max_old = np.max(sigma_a) if np.max(sigma_a) > 0 else 1.0
        shape_new = sigma_a_new / s_max_new
        shape_old = sigma_a / s_max_old
        res_shape = float(np.max(np.abs(shape_new - shape_old)))
        res = max(res_u, res_shape)

        residuals.append(res)
        if verbose:
            print(f"  iter {iteration:4d}  res = {res:.3e}")

        u[:] = u_new
        v[:] = v_new
        sigma_a[:] = sigma_a_next
        sigma_p[:] = sigma_p_next

        if res < tol:
            converged = True
            break

    # Final quantities
    n_e_final = ne_from_sigma_2d(sigma_a, p_pa)
    E_eff_fin = effective_field(np.sqrt(v), p_pa)
    Da_final = ambipolar_diffusion(E_eff_fin, p_pa)
    nu_i_final = ionization_freq(E_eff_fin, p_pa)

    return {
        "r": r,
        "z": z,
        "u": u,
        "v": v,
        "sigma_a": sigma_a,
        "sigma_p": sigma_p,
        "n_e": n_e_final,
        "Da": Da_final,
        "nu_i": nu_i_final,
        "converged": converged,
        "n_iter": len(residuals),
        "residuals": residuals,
    }
