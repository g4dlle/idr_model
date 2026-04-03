"""
test_self_consistent_2d.py — тесты для 2D самосогласованного решателя.

Структура (TDD):
  Phase 1 — compute_lambda0_2d (аналитические инварианты)
  Phase 2 — solve_maxwell_for_ne0_2d (поля и λ₀)
  Phase 3 — find_n_e0_2d с monkeypatch (логика бисекции)
  Phase 4 — интеграционные тесты (медленные)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from equations_2d import make_grid_2d


# ===========================================================================
# Phase 1 — compute_lambda0_2d
# ===========================================================================

class TestComputeLambda0_2D:

    def test_bessel_analytic_neumann(self):
        """
        Da=νi=1, R=1, bc_z="neumann", L большой (→ бесконечная трубка):
        z-зависимость исчезает → λ₀² = j₀₁² ≈ 5.783 (совпадает с 1D).

        Допуск 2%: при Nr=150, Nz=10 и L=10*R дискретизационная погрешность
        по r даёт ~0.5%, по z при Нейман — нулевую.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 1.0, 10.0
        Nr, Nz = 150, 10
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)

        Da   = np.ones((Nr + 1, Nz + 1))
        nu_i = np.ones((Nr + 1, Nz + 1))

        lam0_sq = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="neumann")

        j01 = 2.4048255577
        expected = j01**2  # ≈ 5.783

        assert abs(lam0_sq - expected) / expected < 0.02, (
            f"λ₀² = {lam0_sq:.4f}, ожидалось ≈ {expected:.4f} "
            f"(погрешность {abs(lam0_sq - expected) / expected:.2%})"
        )

    def test_neumann_matches_1d(self):
        """
        При bc_z="neumann" λ₀² 2D совпадает с 1D compute_lambda0
        (задача факторизуется, осевые производные обнуляются).
        """
        from self_consistent_2d import compute_lambda0_2d
        from self_consistent import compute_lambda0
        from equations import make_grid

        R, L = 0.012, 0.05
        Nr, Nz = 80, 20

        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
        r1d, h1d = make_grid(Nr, R)

        Da_val, nu_val = 0.5, 80.0
        Da_2d   = np.full((Nr + 1, Nz + 1), Da_val)
        nu_i_2d = np.full((Nr + 1, Nz + 1), nu_val)
        Da_1d   = np.full(Nr + 1, Da_val)
        nu_i_1d = np.full(Nr + 1, nu_val)

        lam0_sq_2d = compute_lambda0_2d(r, z, hr, hz, Da_2d, nu_i_2d,
                                         bc_z_sigma="neumann")
        lam0_sq_1d = compute_lambda0(r1d, h1d, Da_1d, nu_i_1d)

        assert abs(lam0_sq_2d - lam0_sq_1d) / lam0_sq_1d < 0.02, (
            f"λ₀²_2d={lam0_sq_2d:.4f} vs λ₀²_1d={lam0_sq_1d:.4f} "
            f"(погрешность {abs(lam0_sq_2d - lam0_sq_1d) / lam0_sq_1d:.2%})"
        )

    def test_dirichlet_larger_than_neumann(self):
        """
        Дирихле на торцах добавляет осевую диффузию (π/L)²·Da/νi:
        λ₀²_dirichlet > λ₀²_neumann при любых Da, νi, R, L.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 0.012, 0.05
        Nr, Nz = 60, 20
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)

        Da_val, nu_val = 0.3, 50.0
        Da   = np.full((Nr + 1, Nz + 1), Da_val)
        nu_i = np.full((Nr + 1, Nz + 1), nu_val)

        lam_neu = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="neumann")
        lam_dir = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="dirichlet")

        assert lam_dir > lam_neu, (
            f"Ожидалось λ₀²_dir > λ₀²_neu, получено "
            f"{lam_dir:.4f} vs {lam_neu:.4f}"
        )

    def test_scaled_dirichlet_analytic(self):
        """
        Однородные Da=D, νi=ν, bc_z="dirichlet":
          λ₀² ≈ D · [(j₀₁/R)² + (π/L)²] / ν

        Допуск 3%: слагаемое (π/L)² дискретизируется грубее при малом Nz.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 0.012, 0.05
        Nr, Nz = 100, 40
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)

        D, nu = 0.4, 60.0
        Da   = np.full((Nr + 1, Nz + 1), D)
        nu_i = np.full((Nr + 1, Nz + 1), nu)

        lam0_sq = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="dirichlet")

        j01 = 2.4048255577
        expected = D * ((j01 / R)**2 + (np.pi / L)**2) / nu

        assert abs(lam0_sq - expected) / expected < 0.03, (
            f"λ₀² = {lam0_sq:.4f}, ожидалось {expected:.4f} "
            f"(погрешность {abs(lam0_sq - expected) / expected:.2%})"
        )

    def test_neumann_independent_of_nz(self):
        """
        При bc_z="neumann" λ₀² не зависит от числа z-узлов:
        профиль σ однороден по z, дискретизация по z не влияет.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 0.012, 0.05
        Nr = 60
        Da_val, nu_val = 0.3, 50.0

        results = []
        for Nz in (10, 30):
            r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
            Da   = np.full((Nr + 1, Nz + 1), Da_val)
            nu_i = np.full((Nr + 1, Nz + 1), nu_val)
            results.append(compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                              bc_z_sigma="neumann"))

        assert abs(results[0] - results[1]) / results[0] < 0.01, (
            f"λ₀²(Nz=10)={results[0]:.4f} vs λ₀²(Nz=30)={results[1]:.4f}"
        )

    def test_annular_larger_than_full(self):
        """
        При r_inc > 0 плазма занимает меньший объём →
        эффективный радиус диффузии меньше → λ₀² больше.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 0.012, 0.05
        r_inc = 0.004
        Nr, Nz = 60, 20
        Da_val, nu_val = 0.3, 50.0

        # Полная геометрия
        r_f, z_f, hr_f, hz_f = make_grid_2d(Nr, Nz, R, L, r_min=0.0)
        Da_f   = np.full((Nr + 1, Nz + 1), Da_val)
        nu_f   = np.full((Nr + 1, Nz + 1), nu_val)
        lam_full = compute_lambda0_2d(r_f, z_f, hr_f, hz_f, Da_f, nu_f,
                                      bc_z_sigma="neumann")

        # Аннулярная геометрия
        r_a, z_a, hr_a, hz_a = make_grid_2d(Nr, Nz, R, L, r_min=r_inc)
        Da_a   = np.full((Nr + 1, Nz + 1), Da_val)
        nu_a   = np.full((Nr + 1, Nz + 1), nu_val)
        lam_ann = compute_lambda0_2d(r_a, z_a, hr_a, hz_a, Da_a, nu_a,
                                     bc_z_sigma="neumann")

        assert lam_ann > lam_full, (
            f"Ожидалось λ₀²_ann > λ₀²_full, получено "
            f"{lam_ann:.4f} vs {lam_full:.4f}"
        )


# ===========================================================================
# Phase 2 — solve_maxwell_for_ne0_2d
# ===========================================================================

class TestSolveMaxwell2D:

    def test_wrapper_returns_lambda0_and_valid_fields(self):
        """
        solve_maxwell_for_ne0_2d возвращает словарь с ключом 'lambda0',
        все поля ≥ 0, граничные условия выполнены.
        """
        from self_consistent_2d import solve_maxwell_for_ne0_2d

        result = solve_maxwell_for_ne0_2d(
            n_e0=1e16, Nr=20, Nz=10,
            R=0.012, L=0.05, p_pa=133.0, H_wall=100_000.0,
            max_iter=5, tol=1e-4,
        )

        assert "lambda0" in result, "Нет ключа 'lambda0'"
        assert result["lambda0"] > 0, f"λ₀ ≤ 0: {result['lambda0']}"

        assert np.all(result["u"] >= -1e-10), f"u < 0: min={result['u'].min():.3e}"
        assert np.all(result["v"] >= -1e-10), f"v < 0: min={result['v'].min():.3e}"
        assert np.all(result["sigma_a"] >= -1e-10), "sigma_a < 0"

        # BC: σ = 0 на стенке r = R
        assert np.max(np.abs(result["sigma_a"][-1, :])) < 1e-8, (
            f"σ(R,z) ≠ 0: max={np.max(np.abs(result['sigma_a'][-1, :])):.3e}"
        )

    def test_wrapper_dirichlet_bc_z(self):
        """
        При bc_z_sigma="dirichlet" σ = 0 на обоих торцах.
        """
        from self_consistent_2d import solve_maxwell_for_ne0_2d

        result = solve_maxwell_for_ne0_2d(
            n_e0=1e16, Nr=20, Nz=10,
            R=0.012, L=0.05, p_pa=133.0, H_wall=100_000.0,
            max_iter=5, tol=1e-4, bc_z_sigma="dirichlet",
        )

        assert np.max(np.abs(result["sigma_a"][:, 0])) < 1e-8, "σ(r,0) ≠ 0"
        assert np.max(np.abs(result["sigma_a"][:, -1])) < 1e-8, "σ(r,L) ≠ 0"


# ===========================================================================
# Phase 3 — find_n_e0_2d (monkeypatch: только логика управления)
# ===========================================================================

def _fake_result(lambda0, converged):
    """Минимальный валидный словарь результата для monkeypatch."""
    return {
        "r":        np.linspace(0, 0.012, 5),
        "z":        np.linspace(0, 0.05,  5),
        "u":        np.ones((5, 5)) * 1e10,
        "v":        np.ones((5, 5)) * 1e4,
        "sigma_a":  np.ones((5, 5)) * 0.5,
        "sigma_p":  np.ones((5, 5)) * 0.1,
        "n_e":      np.ones((5, 5)) * 1e16,
        "Da":       np.ones((5, 5)) * 0.3,
        "nu_i":     np.ones((5, 5)) * 50.0,
        "n_iter":   10,
        "residuals": [1e-6],
        "converged": converged,
        "lambda0":  lambda0,
    }


class TestFindNe0_2D:

    def test_bracket_endpoint_exact_root(self, monkeypatch):
        """
        λ₀(n_e0_lo) = 1.0 точно → converged=True, bracket_ok=True.

        Было бы: произведение (1-1)*(10-1)=0 не удовлетворяло бы строгому < 0,
        что приводило к bracket_ok=False.
        """
        import self_consistent_2d as sc2d

        call_count = {"n": 0}

        def fake_solve(n_e0, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _fake_result(lambda0=1.0, converged=True)   # нижняя граница
            if call_count["n"] == 2:
                return _fake_result(lambda0=10.0, converged=True)  # верхняя граница
            return _fake_result(lambda0=5.0, converged=False)

        monkeypatch.setattr(sc2d, "solve_maxwell_for_ne0_2d", fake_solve)

        result = sc2d.find_n_e0_2d(tol_lambda=0.05, max_bisect=10)

        assert result["converged"], (
            f"converged=False при λ₀=1 на границе; "
            f"bracket_ok={result.get('bracket_ok')}"
        )
        assert result.get("bracket_ok"), "bracket_ok должен быть True"
        assert result["solution"] is not None
        assert abs(result["lambda0"] - 1.0) < 0.05

    def test_bracket_fallback_uses_boundary_solution(self, monkeypatch):
        """
        Все внутренние точки не сходятся → solution берётся из граничной,
        а не None (что давало бы E_R=0 в solve_self_consistent_2d).
        """
        import self_consistent_2d as sc2d

        call_count = {"n": 0}

        def fake_solve(n_e0, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _fake_result(lambda0=0.5, converged=True)   # нижняя: |0.5-1|=0.5
            if call_count["n"] == 2:
                return _fake_result(lambda0=5.0, converged=True)   # верхняя: |5-1|=4
            return _fake_result(lambda0=2.0, converged=False)      # внутренние: нет сходимости

        monkeypatch.setattr(sc2d, "solve_maxwell_for_ne0_2d", fake_solve)

        result = sc2d.find_n_e0_2d(tol_lambda=0.05, max_bisect=10)

        assert result["solution"] is not None, (
            "solution=None при наличии сошедшихся граничных точек"
        )
        # Ближайшая к 1: нижняя граница (|0.5-1|=0.5 < |5-1|=4)
        assert abs(result["lambda0"] - 0.5) < 1e-9, (
            f"Ожидался lambda0=0.5, получен {result['lambda0']}"
        )

    def test_no_bracket_returns_bracket_ok_false(self, monkeypatch):
        """
        λ₀ > 1 на обоих концах → bracket_ok=False, converged=False.
        """
        import self_consistent_2d as sc2d

        call_count = {"n": 0}

        def fake_solve(n_e0, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _fake_result(lambda0=2.0, converged=True)
            return _fake_result(lambda0=5.0, converged=True)

        monkeypatch.setattr(sc2d, "solve_maxwell_for_ne0_2d", fake_solve)

        result = sc2d.find_n_e0_2d(tol_lambda=0.05, max_bisect=10)

        assert not result.get("bracket_ok"), "bracket_ok должен быть False"
        assert not result["converged"], "converged должен быть False"


# ===========================================================================
# Phase 4 — интеграционные тесты (медленные)
# ===========================================================================

class TestIntegration2D:

    def test_neumann_bisection_converges(self):
        """
        При bc_z="neumann" бисекция находит скобку и сходится к λ₀≈1.

        Для грубой сетки (Nr=20, Nz=10) корень лежит значительно выше
        1D-оценки (λ₀ растёт медленно по n_e0 при 2D-дискретизации),
        поэтому проверяем только сходимость алгоритма, а не точное
        совпадение n_e0* с 1D.
        """
        from self_consistent_2d import find_n_e0_2d

        res = find_n_e0_2d(
            Nr=20, Nz=10, R=0.012, L=0.05,
            p_pa=133.0, H_wall=100_000.0,
            bc_z_sigma="neumann",
            n_e0_bounds=(1e17, 1e22),   # широкий диапазон для грубой сетки
            tol_lambda=0.15, max_bisect=12,
            solver_kw=dict(max_iter=300, tol=1e-3, relax=0.3),
        )

        assert res.get("bracket_ok"), (
            f"Скобка не найдена: history={res['history'][:4]}"
        )
        assert res["solution"] is not None, "solution=None"
        # λ₀* должна быть в разумной близости от 1
        assert abs(res["lambda0"] - 1.0) < 0.3, (
            f"λ₀*={res['lambda0']:.4f} слишком далеко от 1"
        )

    def test_dirichlet_differs_from_neumann(self):
        """
        λ₀ при bc_z="dirichlet" отличается от "neumann" при том же n_e0:
        Дирихле на торцах добавляет осевую диффузию → λ₀_dir ≠ λ₀_neu.

        Тест не требует полной сходимости бисекции — достаточно сравнить
        λ₀ при фиксированном n_e0 из одного запуска Maxwell.
        """
        from self_consistent_2d import solve_maxwell_for_ne0_2d

        kw = dict(Nr=20, Nz=10, R=0.012, L=0.05,
                  p_pa=133.0, H_wall=100_000.0,
                  max_iter=200, tol=1e-3, relax=0.5)

        # При n_e0=1e21 оба BC сходятся; λ₀ должны отличаться
        r_neu = solve_maxwell_for_ne0_2d(n_e0=1e21, bc_z_sigma="neumann",    **kw)
        r_dir = solve_maxwell_for_ne0_2d(n_e0=1e21, bc_z_sigma="dirichlet",  **kw)

        lam_neu = r_neu["lambda0"]
        lam_dir = r_dir["lambda0"]

        assert abs(lam_neu - lam_dir) / max(lam_neu, lam_dir) > 0.05, (
            f"λ₀ слишком близки: neumann={lam_neu:.4f}, dirichlet={lam_dir:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
