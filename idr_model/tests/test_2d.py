"""
test_2d.py — тесты для 2D осесимметричной модели ИДР.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from equations_2d import make_grid_2d, idx, build_H_equation_2d, build_sigma_equation_2d
from boundary_2d import apply_bc_H_2d, apply_bc_E_2d, apply_bc_sigma_2d
from solver_2d import solve_idr_2d, compute_alpha_2d, ne_from_sigma_2d, compute_E_faraday_2d


# ─── Grid ─────────────────────────────────────────────────────────────────────

class TestGrid2D:
    def test_grid_shape(self):
        r, z, hr, hz = make_grid_2d(10, 20, 0.01, 0.05)
        assert r.shape == (11,)
        assert z.shape == (21,)

    def test_grid_limits(self):
        r, z, hr, hz = make_grid_2d(10, 20, 0.012, 0.05)
        assert r[0] == 0.0
        assert abs(r[-1] - 0.012) < 1e-15
        assert z[0] == 0.0
        assert abs(z[-1] - 0.05) < 1e-15

    def test_grid_spacing(self):
        r, z, hr, hz = make_grid_2d(10, 20, 0.012, 0.05)
        assert abs(hr - 0.012 / 10) < 1e-15
        assert abs(hz - 0.05 / 20) < 1e-15

    def test_grid_with_inclusion(self):
        r, z, hr, hz = make_grid_2d(10, 5, 0.012, 0.05, r_min=0.004)
        assert abs(r[0] - 0.004) < 1e-15
        assert abs(r[-1] - 0.012) < 1e-15

    def test_idx(self):
        Nz1 = 6
        assert idx(0, 0, Nz1) == 0
        assert idx(0, 5, Nz1) == 5
        assert idx(1, 0, Nz1) == 6
        assert idx(2, 3, Nz1) == 15


# ─── Boundary conditions ──────────────────────────────────────────────────────

class TestBoundary2D:
    def test_bc_H_wall(self):
        u = np.ones((5, 6))
        apply_bc_H_2d(u, 100.0)
        np.testing.assert_allclose(u[-1, :], 100.0)

    def test_bc_E_axis(self):
        v = np.ones((5, 6))
        apply_bc_E_2d(v)
        np.testing.assert_allclose(v[0, :], 0.0)

    def test_bc_sigma_wall(self):
        s = np.ones((5, 6))
        apply_bc_sigma_2d(s, bc_z="dirichlet")
        np.testing.assert_allclose(s[-1, :], 0.0)
        np.testing.assert_allclose(s[:, 0], 0.0)
        np.testing.assert_allclose(s[:, -1], 0.0)

    def test_bc_sigma_neumann(self):
        s = np.random.rand(5, 6)
        s_inner = s[:, 1:-1].copy()
        apply_bc_sigma_2d(s, bc_z="neumann")
        # r=R → 0
        np.testing.assert_allclose(s[-1, :], 0.0)
        # z-boundaries → copy of neighbours
        np.testing.assert_allclose(s[:, 0], s[:, 1])
        np.testing.assert_allclose(s[:, -1], s[:, -2])


# ─── Matrix assembly ─────────────────────────────────────────────────────────

class TestMatrixAssembly:
    def test_H_matrix_shape(self):
        Nr, Nz = 5, 4
        r, z, hr, hz = make_grid_2d(Nr, Nz, 0.012, 0.05)
        M = (Nr + 1) * (Nz + 1)
        alpha = np.ones((Nr + 1, Nz + 1))
        sigma_a = np.ones((Nr + 1, Nz + 1))
        v = np.zeros((Nr + 1, Nz + 1))
        A, rhs = build_H_equation_2d(r, z, hr, hz, alpha, sigma_a, v, 1e10)
        assert A.shape == (M, M)
        assert rhs.shape == (M,)

    def test_sigma_matrix_shape(self):
        Nr, Nz = 5, 4
        r, z, hr, hz = make_grid_2d(Nr, Nz, 0.012, 0.05)
        M = (Nr + 1) * (Nz + 1)
        Da = np.ones((Nr + 1, Nz + 1)) * 1e-3
        nu_i = np.ones((Nr + 1, Nz + 1)) * 100.0
        sigma_ref = np.ones((Nr + 1, Nz + 1))
        A, rhs = build_sigma_equation_2d(r, z, hr, hz, Da, nu_i,
                                          sigma_ref=sigma_ref)
        assert A.shape == (M, M)
        assert rhs.shape == (M,)


# ─── Solver ───────────────────────────────────────────────────────────────────

class TestSolver2D:
    def test_solver_runs(self):
        """Solver runs without crashing with small grid."""
        result = solve_idr_2d(
            Nr=10, Nz=5, R=0.012, L=0.05,
            p_pa=133.0, H_wall=100000.0,
            max_iter=5, tol=1e-3, relax=0.5,
        )
        assert "u" in result
        assert "v" in result
        assert "sigma_a" in result
        assert result["u"].shape == (11, 6)
        assert result["v"].shape == (11, 6)
        assert result["n_iter"] == 5 or result["converged"]

    def test_fields_nonnegative(self):
        """All fields must be >= 0."""
        result = solve_idr_2d(
            Nr=10, Nz=5, R=0.012, L=0.05,
            p_pa=133.0, H_wall=100000.0,
            max_iter=20, tol=1e-3, relax=0.5,
        )
        assert np.all(result["u"] >= 0)
        assert np.all(result["v"] >= 0)
        assert np.all(result["sigma_a"] >= 0)
        assert np.all(result["n_e"] >= 0)

    def test_bc_satisfied(self):
        """Boundary conditions must hold in the solution."""
        result = solve_idr_2d(
            Nr=10, Nz=5, R=0.012, L=0.05,
            p_pa=133.0, H_wall=100000.0,
            max_iter=20, tol=1e-3, relax=0.5,
        )
        # u(R) = H_wall²
        np.testing.assert_allclose(result["u"][-1, :], 100000.0**2, rtol=1e-10)
        # v(0) = 0
        np.testing.assert_allclose(result["v"][0, :], 0.0, atol=1e-10)
        # sigma(R) = 0
        np.testing.assert_allclose(result["sigma_a"][-1, :], 0.0, atol=1e-10)

    def test_neumann_z_uniform(self):
        """With Neumann BCs in z, the solution should be nearly z-uniform."""
        result = solve_idr_2d(
            Nr=15, Nz=8, R=0.012, L=0.05,
            p_pa=133.0, H_wall=100000.0,
            max_iter=50, tol=1e-4, relax=0.5,
            bc_z_sigma="neumann",
        )
        # Check that sigma varies less than 5% along z at the midplane of r
        i_mid = len(result["r"]) // 2
        sigma_z = result["sigma_a"][i_mid, :]
        if sigma_z.max() > 0:
            variation = (sigma_z.max() - sigma_z.min()) / sigma_z.max()
            assert variation < 0.05, f"z-variation too high: {variation:.3f}"


# ─── Helper functions ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_alpha_positive(self):
        sa = np.array([1.0, 2.0, 0.0])
        sp = np.array([0.5, 1.0, 0.0])
        alpha = compute_alpha_2d(sa, sp)
        assert np.all(alpha >= 0)

    def test_ne_roundtrip(self):
        """conductivity → ne_from_sigma should give back n_e."""
        from physics import conductivity
        n_e = 1e16
        p_pa = 133.0
        sa, sp, _ = conductivity(n_e, p_pa)
        n_e_back = ne_from_sigma_2d(sa, p_pa)
        np.testing.assert_allclose(n_e_back, n_e, rtol=1e-10)
