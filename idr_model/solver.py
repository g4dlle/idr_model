"""
solver.py -- iterative solver for the IDR system.

Algorithm: sequential Gauss-Seidel over equations.
Each equation is solved by the Thomas (tridiagonal) algorithm.

Iteration order:
  1. Solve for |H|^2  (sigma known from previous iteration)
  2. Compute |E_phi|^2 from Faraday's law (integral form):
       E_phi(r) = omega*mu0/r * integral_0^r H(r')*r' dr'
     This gives the physically correct azimuthal electric field.
  3. Compute E_eff(r) from |E|^2
  4. Update Da(r), nu_i(r)
  5. Solve for sigma(r)
  6. Under-relaxation: sigma_new = w*sigma_new + (1-w)*sigma_old
  7. Check convergence: max|Delta_sigma/sigma| < tol
"""

import numpy as np
from equations import (
    make_grid, build_H_equation, build_sigma_equation,
)
from physics import (
    conductivity, sigma_from_conductivity,
    effective_field, ambipolar_diffusion, ionization_freq,
)
from boundary import apply_wall_sigma
from config import (
    N_GRID, R_TUBE, H_WALL, P_PA, N_E0,
    MAX_ITER, TOL, RELAX,
    E_CHARGE, M_ELECTRON, OMEGA,
    NU_C_PER_TORR, DT,
)

MU_0 = 4.0 * np.pi * 1e-7   # H/m, vacuum permeability


# ---------------------------------------------------------------------------
# Thomas algorithm (tridiagonal solver)
# ---------------------------------------------------------------------------

def thomas_solve(lower, main, upper, rhs):
    """
    Solve tridiagonal system by Thomas algorithm.

    System:  lower[i]*x[i-1] + main[i]*x[i] + upper[i]*x[i+1] = rhs[i]

    lower[0] and upper[N] are ignored.

    Parameters
    ----------
    lower, main, upper, rhs : arrays of length N+1

    Returns
    -------
    x : solution array (N+1,)
    """
    n = len(main)
    c = upper.copy().astype(float)
    d = rhs.copy().astype(float)
    m = main.copy().astype(float)

    if abs(m[0]) < 1e-300:
        raise ValueError("Zero pivot in Thomas algorithm at i=0")
    c[0] /= m[0]
    d[0] /= m[0]

    for i in range(1, n):
        denom = m[i] - lower[i] * c[i - 1]
        if abs(denom) < 1e-300:
            raise ValueError(f"Zero pivot in Thomas algorithm at i={i}")
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (d[i] - lower[i] * d[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


# ---------------------------------------------------------------------------
# Helper computations on the grid
# ---------------------------------------------------------------------------

def ne_from_sigma(sigma_a, p_pa):
    """
    Extract electron density from active conductivity.

    sigma_a = n_e*e^2*nu_c / (m_e*(nu_c^2+omega^2))
    => n_e = sigma_a * m_e*(nu_c^2+omega^2) / (e^2*nu_c)
    """
    from physics import collision_freq
    nu_c = collision_freq(p_pa)
    return sigma_a * M_ELECTRON * (nu_c**2 + OMEGA**2) / (E_CHARGE**2 * nu_c)


def compute_alpha(sigma_a, sigma_p):
    """alpha = sigma_a / |sigma|^2"""
    mod2 = sigma_a**2 + sigma_p**2
    safe = np.where(mod2 > 0, mod2, 1e-300)
    return sigma_a / safe


# ---------------------------------------------------------------------------
# Main iterative solver
# ---------------------------------------------------------------------------

def solve_idr(N=N_GRID, R=R_TUBE, p_pa=P_PA, H_wall=H_WALL, n_e0=N_E0,
              max_iter=MAX_ITER, tol=TOL, relax=RELAX, dt=None,
              r_inc=0.0, verbose=False):
    """
    Iterative solver for the 1D IDR model.

    Parameters
    ----------
    N        : number of grid intervals
    R        : tube radius [m]
    p_pa     : pressure [Pa]
    H_wall   : H-field amplitude at the wall [A/m]
    n_e0     : initial electron density [m^-3]
    max_iter : maximum iteration count
    tol      : convergence criterion  max|Δσ/σ|
    relax    : under-relaxation for H field (0 < relax <= 1)
    dt       : pseudo-time step [s] for sigma equation.
               None  → power iteration + self-consistent normalisation (default).
               float → IMEX time-stepping.
    r_inc    : inner radius of conducting inclusion [m].
               0.0   → standard geometry (full cylinder, axis at r=0).
               > 0   → annular geometry r ∈ [r_inc, R].
    verbose  : print residuals if True

    Returns
    -------
    dict with keys: r, u (|H|^2), v (|E|^2), sigma_a, sigma_p,
                    n_e, Da, nu_i, converged, n_iter, residuals
    """
    r, h = make_grid(N, R, r_min=r_inc)
    annular = r_inc > 0  # кольцевая геометрия

    # Initialisation
    sigma_a0, sigma_p0, _ = conductivity(n_e0, p_pa)
    if annular:
        # Парабола в [r_inc, R] с нулями на обеих стенках
        profile = (r - r_inc) * (R - r) / ((R - r_inc) / 2.0)**2
        profile = np.maximum(profile, 0.0)
    else:
        # Bessel-like initial profile: sigma ~ (1 - (r/R)^2)
        profile = np.maximum(1.0 - (r / R)**2, 0.0)
    sigma_a = sigma_a0 * profile
    sigma_a[0] = 0.0 if annular else sigma_a[0]
    sigma_a[-1] = 0.0
    sigma_p = sigma_p0 * profile
    sigma_p[0] = 0.0 if annular else sigma_p[0]
    sigma_p[-1] = 0.0

    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    u[-1] = H_wall**2

    residuals = []
    converged = False

    for iteration in range(max_iter):

        # Step a: solve for |H|^2
        alpha = compute_alpha(sigma_a, sigma_p)
        # Guard: avoid zero alpha everywhere
        if np.all(alpha == 0):
            alpha = np.full(N + 1, 1e-10)

        l, m, up, rhs = build_H_equation(r, h, alpha, sigma_a, v, H_wall**2)
        u_new = thomas_solve(l, m, up, rhs)
        u_new = np.maximum(u_new, 0.0)

        # Step b: compute |E_phi|^2 from Faraday's law (integral form)
        # E_phi(r) = omega*mu0/r * integral_{r_start}^r H(r')*r' dr'
        # r_start = r_inc (annular) or 0 (full cylinder)
        # At r_start: E_phi = 0 (conductor surface or axis)
        H_abs = np.sqrt(np.maximum(u_new, 0.0))
        integrand = H_abs * r           # H(r') * r'
        # cumulative trapezoidal integral on uniform grid
        intervals = 0.5 * h * (integrand[:-1] + integrand[1:])
        integrals = np.zeros(N + 1)
        integrals[1:] = np.cumsum(intervals)
        # v[0] = 0 (E_phi = 0 at r_inc or axis)
        v_new = np.zeros(N + 1)
        v_new[1:] = (OMEGA * MU_0 * integrals[1:] / r[1:])**2

        # Step c: effective field
        E_abs = np.sqrt(v_new)
        E_eff = effective_field(E_abs, p_pa)

        # Step d: transport coefficients
        Da   = ambipolar_diffusion(E_eff, p_pa)
        nu_i = ionization_freq(E_eff, p_pa)

        # Step e: solve for sigma profile
        if dt is not None:
            # ── Режим псевдоустановления по времени (IMEX) ────────────────────
            # (σ_new − σ_old)/dt = L[σ_new] + νi·σ_old
            # Диффузия неявная (устойчиво), ионизация явная.
            # Амплитуда n_e эволюционирует к физическому равновесию νi = λ₁·Da
            # без какой-либо внешней нормировки.
            l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i,
                                                  sigma_a_ref=sigma_a, dt=dt)
            sigma_a_new = thomas_solve(l, m, up, rhs)
            sigma_a_new = np.maximum(sigma_a_new, 0.0)
            if annular:
                sigma_a_new[0] = 0.0   # рекомбинация на включении
            apply_wall_sigma(sigma_a_new)
            _, sigma_p_new, _ = conductivity(ne_from_sigma(sigma_a_new, p_pa), p_pa)
            # Нет под-релаксации: шаг dt уже контролирует скорость изменения
            sigma_a_next = sigma_a_new
            sigma_p_next = sigma_p_new
        else:
            # ── Итерация по мощности + самосогласованная нормировка ──────────
            # (-Da·Δ)·σ_new = νi·σ_old  →  сходится к J₀ (форма профиля)
            # Амплитуда нормируется к текущей n_e (не зафиксирована на n_e0).
            #
            # Цепочка единиц:
            #   sigma_raw  [S/m];  ne_from_sigma → n_e_current [m⁻³]
            #   integral_ref = ∫ n_e_current · r dr  [m⁻¹]
            #   integral_new = ∫ sigma_raw   · r dr  [S·m]
            #   n_e_new = sigma_raw · integral_ref / integral_new  [m⁻³] ✓
            l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i, sigma_a_ref=sigma_a)
            sigma_raw = thomas_solve(l, m, up, rhs)
            sigma_raw = np.maximum(sigma_raw, 0.0)
            if annular:
                sigma_raw[0] = 0.0     # рекомбинация на включении
            apply_wall_sigma(sigma_raw)

            n_e_raw = sigma_raw
            n_e_raw[-1] = 0.0

            n_e_current = ne_from_sigma(sigma_a, p_pa)                         # [m⁻³]
            integral_ref = np.trapezoid(n_e_current[:-1] * r[:-1], r[:-1])    # [m⁻¹]
            target_dyn   = integral_ref if integral_ref > 1e-300 else n_e0 * R**2 / 2.0

            integral_new = np.trapezoid(n_e_raw[:-1] * r[:-1], r[:-1])        # [S·m]
            norm = integral_new if integral_new > 1e-300 else 1.0
            n_e_new = n_e_raw * target_dyn / norm    # [m⁻³]
            n_e_new[-1] = 0.0

            sigma_a_new, sigma_p_new, _ = conductivity(n_e_new, p_pa)

            # Под-релаксация (стабилизирует итерации по мощности)
            sigma_a_next = relax * sigma_a_new + (1.0 - relax) * sigma_a
            sigma_p_next = relax * sigma_p_new + (1.0 - relax) * sigma_p

        # Step g: convergence check.
        if dt is not None:
            # ── Режим псевдоустановления ──────────────────────────────────────
            # Критерий: относительное изменение σ между шагами по времени.
            # max|σ_new − σ_old| / max(σ_new) → 0 при σ_new = σ_old (равновесие).
            # Нормировка по σ_max чувствительна и к изменению АМПЛИТУДЫ,
            # и к изменению ФОРМЫ профиля.
            s_scale = max(np.max(sigma_a_next[:-1]), 1e-300)
            res = float(np.max(np.abs(sigma_a_next[:-1] - sigma_a[:-1])) / s_scale)
        else:
            # ── Режим итерации по мощности ────────────────────────────────────
            # (a) Относительное изменение поля |H|²
            if np.any(u[:-1] > 0):
                safe_u = np.where(np.abs(u[:-1]) > 0, np.abs(u[:-1]), 1e-300)
                res_u = float(np.max(np.abs(u_new[:-1] - u[:-1]) / safe_u))
            else:
                res_u = 1.0
            # (b) Изменение нормированной формы профиля σ (независимо от амплитуды,
            #     которая фиксирована нормировкой — нет смысла её отслеживать)
            s_max_new = np.max(sigma_a_new[:-1]) if np.max(sigma_a_new[:-1]) > 0 else 1.0
            s_max_old = np.max(sigma_a[:-1])     if np.max(sigma_a[:-1]) > 0 else 1.0
            shape_new = sigma_a_new[:-1] / s_max_new
            shape_old = sigma_a[:-1]     / s_max_old
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

    # Final derived quantities
    n_e_final  = ne_from_sigma(sigma_a, p_pa)
    E_eff_fin  = effective_field(np.sqrt(v), p_pa)
    Da_final   = ambipolar_diffusion(E_eff_fin, p_pa)
    nu_i_final = ionization_freq(E_eff_fin, p_pa)

    return {
        "r":        r,
        "u":        u,
        "v":        v,
        "sigma_a":  sigma_a,
        "sigma_p":  sigma_p,
        "n_e":      n_e_final,
        "Da":       Da_final,
        "nu_i":     nu_i_final,
        "converged": converged,
        "n_iter":   len(residuals),
        "residuals": residuals,
    }
