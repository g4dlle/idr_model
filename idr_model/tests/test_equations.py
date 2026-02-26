"""
test_equations.py -- tests for equations.py module.

Tests 2.1-2.6 from the implementation plan.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from equations import (
    make_grid, build_H_equation, build_E_equation,
    build_sigma_equation, half_node_avg,
)
from solver import thomas_solve


# --- 2.1 Constant sigma: operator reproduces the exact solution ---------------
def test_H_equation_constant_sigma():
    """
    With sigma_a=const, sigma_p=const, v=0 equation (11) reduces to
    (1/r) d/dr(r*du/dr) = 0 => u = const (with u'(0)=0 and Dirichlet at R).
    """
    N = 50
    R = 0.01
    r, h = make_grid(N, R)

    sigma_a = np.ones(N + 1) * 1.0
    sigma_p = np.ones(N + 1) * 0.5
    alpha   = sigma_a / (sigma_a**2 + sigma_p**2)
    v       = np.zeros(N + 1)   # no source
    H_wall_sq = 4.0

    l, m, up, rhs = build_H_equation(r, h, alpha, sigma_a, v, H_wall_sq)
    u = thomas_solve(l, m, up, rhs)

    # BC must be satisfied exactly
    assert abs(u[-1] - H_wall_sq) < 1e-12, f"u[N]={u[-1]}, expect {H_wall_sq}"

    # Solution must be constant = H_wall_sq (no source, symmetric BC)
    assert np.allclose(u, H_wall_sq, atol=1e-8), (
        f"u not constant: min={u.min():.6f}, max={u.max():.6f}"
    )


# --- 2.2 Diffusion operator: r-weighted symmetry -----------------------------
def test_diffusion_operator_symmetric():
    """
    The cylindrical diffusion operator L[u] = (1/r)*d/dr(r*Da*du/dr) with
    Da=const is self-adjoint in the weighted inner product <f,g>_r = int f*g*r dr.

    This means the matrix W^{1/2} A W^{1/2} is symmetric, where W = diag(r_i * h).

    Equivalently: A[i,j] * r[i] = A[j,i] * r[j]  for internal nodes.

    We verify this for the sigma equation (13) with Da=const, nu_i=0.
    """
    N = 20
    R = 0.01
    r, h = make_grid(N, R)

    Da   = np.ones(N + 1) * 1.0
    nu_i = np.zeros(N + 1)

    l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i)

    # Build full matrix (internal rows only)
    A_full = np.diag(m) + np.diag(l[1:], -1) + np.diag(up[:-1], 1)
    A_inner = A_full[1:N, 1:N]
    r_inner = r[1:N]

    # Weighted matrix: B[i,j] = A_inner[i,j] * r_inner[i]
    # For self-adjointness: B = B^T  (i.e. A[i,j]*r[i] = A[j,i]*r[j])
    B = A_inner * r_inner[:, np.newaxis]
    assert np.allclose(B, B.T, rtol=1e-6, atol=1e-10), (
        "Cylindrical diffusion operator is not r-weighted symmetric.\n"
        f"Max asymmetry: {np.max(np.abs(B - B.T)):.3e}"
    )


# --- 2.3 Zero RHS, zero source => constant solution --------------------------
def test_zero_rhs_constant_solution():
    """
    Equation (13) with nu_i=0, RHS=0, BC: d_sigma/dr(0)=0, sigma(R)=C => sigma=C.
    """
    N = 40
    R = 0.01
    r, h = make_grid(N, R)

    Da   = np.ones(N + 1) * 1.0
    nu_i = np.zeros(N + 1)

    l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i)

    # BC: sigma(R) = 1
    l[N], m[N], up[N], rhs[N] = 0.0, 1.0, 0.0, 1.0

    sigma = thomas_solve(l, m, up, rhs)

    assert np.allclose(sigma, 1.0, atol=1e-10), (
        f"sigma not constant: min={sigma.min():.6f}, max={sigma.max():.6f}"
    )


# --- 2.4 Flux conservation ---------------------------------------------------
def test_flux_conservation():
    """
    For equation (13) without source (nu_i=0):
    The discrete residual A*x - b = 0 for the computed solution.
    """
    N = 40
    R = 0.01
    r, h = make_grid(N, R)

    Da   = np.ones(N + 1) * 2.0
    nu_i = np.zeros(N + 1)

    l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i)
    l[N], m[N], up[N], rhs[N] = 0.0, 1.0, 0.0, 0.0

    sigma = thomas_solve(l, m, up, rhs)

    # Check residual for internal nodes
    residual = np.zeros(N + 1)
    for i in range(1, N):
        residual[i] = (l[i] * sigma[i-1] + m[i] * sigma[i]
                       + up[i] * sigma[i+1] - rhs[i])

    assert np.max(np.abs(residual[1:N])) < 1e-10, (
        f"Max residual = {np.max(np.abs(residual[1:N])):.3e}"
    )


# --- 2.5 Second-order convergence O(h^2) ------------------------------------
def test_grid_convergence_order2():
    """
    For the cylindrical diffusion equation with known exact solution,
    the scheme has second-order convergence: error ~ h^2.

    Test problem: (1/r) d/dr(r * du/dr) = f, u(R)=0, u'(0)=0.

    Exact solution: u = A * (1 - (r/R)^2), A arbitrary.
    => (1/r) d/dr(r * du/dr) = (1/r) d/dr(-2*A*r/R^2) = -4*A/R^2 = const.

    We set A=1 => u_exact = 1 - (r/R)^2, f = -4/R^2.
    Boundary condition: u(R) = 0 (Dirichlet).
    """
    R = 0.01
    f_val = -4.0 / R**2

    def u_exact(r):
        return 1.0 - (r / R)**2

    def solve_for_N(N_pts):
        r, h = make_grid(N_pts, R)
        # Use sigma equation with Da=1, nu_i=0 (pure Laplacian)
        # and add a uniform RHS f_val through a source term.
        Da   = np.ones(N_pts + 1)
        nu_i = np.zeros(N_pts + 1)

        l, m, up, rhs_base = build_sigma_equation(r, h, Da, nu_i)

        # The sigma equation discretises -L[u] = 0 (matrix A = -L).
        # A*u_exact = -f_val (since L[u_exact] = f_val = -4/R^2 < 0).
        # So to solve L[u] = f_val we set rhs = -f_val.
        rhs_base[0] = -f_val
        for i in range(1, N_pts):
            rhs_base[i] = -f_val

        # BC: u(R) = 0
        l[N_pts], m[N_pts], up[N_pts], rhs_base[N_pts] = 0.0, 1.0, 0.0, 0.0

        u = thomas_solve(l, m, up, rhs_base)
        err = np.max(np.abs(u - u_exact(r)))
        return err

    err1 = solve_for_N(50)
    err2 = solve_for_N(100)

    ratio = err1 / (err2 + 1e-300)
    assert ratio > 3.5, (
        f"Convergence order < 2: errors {err1:.3e}/{err2:.3e} = {ratio:.2f} (expected ~4)"
    )


# --- 2.6 Regularity at singular point r=0 ------------------------------------
def test_singular_point_regularity():
    """
    At r=0, the L'Hopital rule is used:
    (1/r)*d/dr(r*f') -> 2*f'' as r->0.

    Verify that the solution is smooth (symmetric) at the axis:
    du/dr ~ 0 at r=0.
    """
    N = 100
    R = 0.01
    r, h = make_grid(N, R)

    alpha   = np.ones(N + 1)
    sigma_a = np.ones(N + 1) * 0.1
    v       = np.ones(N + 1) * 0.5
    H_wall_sq = 1.0

    l, m, up, rhs = build_H_equation(r, h, alpha, sigma_a, v, H_wall_sq)
    u = thomas_solve(l, m, up, rhs)

    # Numerical derivative at axis
    du_axis = abs((u[1] - u[0]) / h)
    # Derivative in the middle
    mid = N // 2
    du_mid = abs((u[mid+1] - u[mid]) / h)

    # At the axis the gradient must be much smaller than in the interior
    assert du_axis < du_mid * 0.5 or du_axis < 1e-3, (
        f"Non-zero gradient at axis: du/dr|_0 = {du_axis:.4e}, "
        f"du/dr|_mid = {du_mid:.4e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
