"""
test_inclusion.py — тесты для модели ИДР с проводящим цилиндрическим включением.

Проводящее включение радиуса r_inc расположено на оси, область разряда —
кольцо r ∈ [r_inc, R].

Граничные условия на поверхности проводника (r = r_inc):
  • E_φ = 0  →  v[0] = 0   (поле не проникает внутрь идеального проводника)
  • σ = 0               (рекомбинация на проводящей поверхности)
  • dH/dr = 0           (по умолчанию, Неймана — касательная H непрерывна)

Граничные условия на стенке (r = R):
  • |H|² = H_wall²      (Дирихле)
  • σ = 0               (рекомбинация)

Группы тестов:
  T1 — Сетка: make_grid с r_min > 0
  T2 — Граничные условия на включении
  T3 — Уравнения: build_*_equation для кольцевой геометрии
  T4 — Решатель: solve_idr с r_inc > 0
  T5 — Интеграционные: физические свойства решения
  T6 — Совместимость: r_inc=0 ≡ без включения
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# T1: Сетка с внутренним радиусом
# ═══════════════════════════════════════════════════════════════════════════════

class TestGridWithInclusion:
    """Тесты для make_grid с параметром r_min > 0."""

    def test_grid_starts_at_r_inc(self):
        """Сетка r ∈ [r_inc, R] при r_min = r_inc."""
        from equations import make_grid

        N = 50
        R = 0.012
        r_inc = 0.004

        r, h = make_grid(N, R, r_min=r_inc)

        assert len(r) == N + 1, f"Длина сетки {len(r)} != {N+1}"
        assert abs(r[0] - r_inc) < 1e-15, f"r[0]={r[0]}, ожидалось {r_inc}"
        assert abs(r[-1] - R) < 1e-15, f"r[-1]={r[-1]}, ожидалось {R}"

    def test_grid_uniform_spacing(self):
        """Шаг h = (R - r_inc) / N на кольцевой сетке."""
        from equations import make_grid

        N = 100
        R = 0.012
        r_inc = 0.003

        r, h = make_grid(N, R, r_min=r_inc)

        expected_h = (R - r_inc) / N
        assert abs(h - expected_h) < 1e-15, f"h={h}, ожидалось {expected_h}"

        # Все интервалы одинаковые
        diffs = np.diff(r)
        assert np.allclose(diffs, h, atol=1e-14), "Неравномерная сетка"

    def test_grid_r_min_zero_unchanged(self):
        """При r_min=0 (дефолт) сетка не меняется."""
        from equations import make_grid

        N = 50
        R = 0.012
        r_default, h_default = make_grid(N, R)
        r_zero, h_zero = make_grid(N, R, r_min=0.0)

        assert np.array_equal(r_default, r_zero)
        assert h_default == h_zero


# ═══════════════════════════════════════════════════════════════════════════════
# T2: Граничные условия на включении
# ═══════════════════════════════════════════════════════════════════════════════

class TestInclusionBoundaryConditions:
    """Тесты для ГУ на внутренней стенке проводящего включения."""

    def test_apply_inclusion_E(self):
        """E_φ = 0 на поверхности проводника: v[0] = 0."""
        from boundary import apply_inclusion_E

        v = np.array([3.14, 1.0, 2.0, 3.0, 4.0])
        apply_inclusion_E(v)
        assert v[0] == 0.0, f"v[0]={v[0]}, ожидалось 0.0"
        # Остальные элементы не затронуты
        assert v[1] == 1.0 and v[2] == 2.0

    def test_apply_inclusion_sigma(self):
        """σ = 0 на поверхности проводника: sigma[0] = 0."""
        from boundary import apply_inclusion_sigma

        sigma = np.ones(10) * 5.0
        apply_inclusion_sigma(sigma)
        assert sigma[0] == 0.0, f"sigma[0]={sigma[0]}, ожидалось 0.0"
        assert np.all(sigma[1:] == 5.0), "Затронуты внутренние узлы"

    def test_apply_inclusion_H_neumann(self):
        """dH/dr = 0 на включении (по умолчанию, H_inc=None)."""
        from boundary import apply_inclusion_H

        u = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        # При H_inc=None функция не должна менять u[0]
        # (ГУ Неймана — обрабатывается в уравнении)
        u_orig_0 = u[0]
        apply_inclusion_H(u, H_inc=None)
        assert u[0] == u_orig_0, "Неймана: u[0] не должен меняться"

    def test_apply_inclusion_H_dirichlet(self):
        """|H|² = H_inc² на включении при заданном H_inc."""
        from boundary import apply_inclusion_H

        u = np.zeros(10)
        H_inc = 500.0
        apply_inclusion_H(u, H_inc=H_inc)
        assert u[0] == H_inc**2, f"u[0]={u[0]}, ожидалось {H_inc**2}"


# ═══════════════════════════════════════════════════════════════════════════════
# T3: Уравнения для кольцевой геометрии
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquationsAnnular:
    """Тесты дискретизации уравнений на кольцевой сетке r ∈ [r_inc, R]."""

    def test_H_equation_constant_sigma_annular(self):
        """
        Уравнение (11) с σ=const, v=0 на кольцевой сетке:
        (1/r)d/dr(r·du/dr) = 0 с ГУ u'(r_inc)=0, u(R) = H_wall².
        Решение: u = const = H_wall².
        """
        from equations import make_grid, build_H_equation
        from solver import thomas_solve

        N = 80
        R = 0.012
        r_inc = 0.004
        r, h = make_grid(N, R, r_min=r_inc)

        sigma_a = np.ones(N + 1) * 1.0
        sigma_p = np.ones(N + 1) * 0.5
        alpha = sigma_a / (sigma_a**2 + sigma_p**2)
        v = np.zeros(N + 1)
        H_wall_sq = 4.0

        l, m, up, rhs = build_H_equation(r, h, alpha, sigma_a, v, H_wall_sq)
        u = thomas_solve(l, m, up, rhs)

        # Граничное условие
        assert abs(u[-1] - H_wall_sq) < 1e-12, f"u[N]={u[-1]}, ожидалось {H_wall_sq}"

        # Решение — константа (нет источника, Неймана + Дирихле)
        assert np.allclose(u, H_wall_sq, atol=1e-8), (
            f"u не константа: min={u.min():.6f}, max={u.max():.6f}"
        )

    def test_E_equation_dirichlet_at_inclusion(self):
        """
        Уравнение (12) на кольцевой сетке: ГУ v[0] = 0 (E=0 на проводнике).
        """
        from equations import make_grid, build_E_equation
        from solver import thomas_solve

        N = 80
        R = 0.012
        r_inc = 0.004
        r, h = make_grid(N, R, r_min=r_inc)

        alpha = np.ones(N + 1) * 0.5
        sigma_a = np.ones(N + 1) * 0.1
        u = np.ones(N + 1)

        l, m, up, rhs = build_E_equation(r, h, alpha, sigma_a, u)
        v = thomas_solve(l, m, up, rhs)

        # ГУ Дирихле: v[0] = 0
        assert abs(v[0]) < 1e-12, f"v[0]={v[0]:.3e}, ожидалось 0"

    def test_sigma_equation_zero_on_both_walls(self):
        """
        Уравнение (13) на кольцевой сетке: σ(r_inc) = 0 и σ(R) = 0.
        """
        from equations import make_grid, build_sigma_equation
        from solver import thomas_solve

        N = 80
        R = 0.012
        r_inc = 0.004
        r, h = make_grid(N, R, r_min=r_inc)

        Da = np.ones(N + 1) * 1.0
        nu_i = np.ones(N + 1) * 100.0

        l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i)
        sigma = thomas_solve(l, m, up, rhs)

        # ГУ Дирихле на обеих стенках
        assert abs(sigma[0]) < 1e-12, f"σ(r_inc)={sigma[0]:.3e} ≠ 0"
        assert abs(sigma[-1]) < 1e-12, f"σ(R)={sigma[-1]:.3e} ≠ 0"

    def test_sigma_profile_annular_positive(self):
        """
        При Da=const, νi=const > 0 и σ=0 на обеих стенках,
        решение σ(r) ≥ 0 внутри кольца.
        """
        from equations import make_grid, build_sigma_equation
        from solver import thomas_solve

        N = 100
        R = 0.012
        r_inc = 0.003
        r, h = make_grid(N, R, r_min=r_inc)

        Da = np.ones(N + 1) * 1.0
        nu_i = np.ones(N + 1) * 500.0

        # Power iteration с σ_ref = парабола
        sigma_ref = (r - r_inc) * (R - r) / ((R - r_inc) / 2)**2
        l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i, sigma_a_ref=sigma_ref)
        sigma = thomas_solve(l, m, up, rhs)

        # Профиль неотрицателен
        assert np.all(sigma >= -1e-10), (
            f"σ < 0: min={sigma.min():.3e}"
        )

    def test_grid_convergence_annular(self):
        """
        Схема O(h²): ошибка при N=100 должна быть ≈ 4× меньше, чем при N=50.

        Тестовая задача на кольце: (1/r)d/dr(r·du/dr) = f, u(r_inc)=0, u(R)=0.
        Точное решение: u = (r² - R²)/4 - (r_inc² - R²)/4 · ln(r/R) / ln(r_inc/R).
        Упрощённо: u = A·(1 - (r/R)²) + B·ln(r/R).
        """
        from equations import make_grid, build_sigma_equation
        from solver import thomas_solve

        R = 0.012
        r_inc = 0.004

        def solve_for_N(N_pts):
            r, h = make_grid(N_pts, R, r_min=r_inc)
            # Задача: -Da·Δu = f, Da=1, f = const = 4/R²
            Da = np.ones(N_pts + 1)
            nu_i = np.zeros(N_pts + 1)

            l, m, up, rhs_base = build_sigma_equation(r, h, Da, nu_i)

            # RHS = f = -4/R² (знак: уравнение записано как Da·Δu - νi·σ = 0
            # → -Δu = f/Da → rhs = f)
            f_val = -4.0 / R**2
            for i in range(N_pts + 1):
                if i == 0 or i == N_pts:
                    continue
                rhs_base[i] = -f_val

            # ГУ: u(r_inc) = 0, u(R) = 0
            l[0], m[0], up[0], rhs_base[0] = 0.0, 1.0, 0.0, 0.0
            l[N_pts], m[N_pts], up[N_pts], rhs_base[N_pts] = 0.0, 1.0, 0.0, 0.0

            u = thomas_solve(l, m, up, rhs_base)
            return r, u

        r1, u1 = solve_for_N(50)
        r2, u2 = solve_for_N(100)

        # Интерполируем грубое решение на тонкую сетку
        u1_interp = np.interp(r2, r1, u1)
        diff = np.max(np.abs(u1_interp - u2))

        r3, u3 = solve_for_N(200)
        u2_interp = np.interp(r3, r2, u2)
        diff2 = np.max(np.abs(u2_interp - u3))

        # Отношение ошибок ~ 4 при h → h/2 (второй порядок)
        if diff2 > 1e-15:
            ratio = diff / diff2
            assert ratio > 3.0, (
                f"Порядок сходимости < 2: ошибки {diff:.3e}/{diff2:.3e} = {ratio:.2f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# T4: Решатель с включением
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolverWithInclusion:
    """Тесты итерационного решателя solve_idr с r_inc > 0."""

    def test_solver_converges(self):
        """solve_idr с r_inc > 0 сходится за разумное число итераций."""
        from solver import solve_idr

        result = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        assert result["converged"], (
            f"Решатель не сошёлся за {result['n_iter']} итераций"
        )

    def test_solver_boundary_conditions_satisfied(self):
        """Все ГУ выполнены в сошедшемся решении."""
        from solver import solve_idr

        H_wall = 100000.0
        r_inc = 0.004
        result = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=H_wall,
            r_inc=r_inc,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        u = result["u"]
        v = result["v"]
        sigma_a = result["sigma_a"]

        # u(R) = H_wall²
        assert abs(u[-1] - H_wall**2) < 1e-6, (
            f"|H|²(R) = {u[-1]:.3e}, ожидалось {H_wall**2:.3e}"
        )

        # v(r_inc) = 0 (E=0 на проводнике)
        assert abs(v[0]) < 1e-6, f"|E|²(r_inc) = {v[0]:.3e} ≠ 0"

        # σ(r_inc) = 0
        assert abs(sigma_a[0]) < 1e-8, f"σ(r_inc) = {sigma_a[0]:.3e} ≠ 0"

        # σ(R) = 0
        assert abs(sigma_a[-1]) < 1e-8, f"σ(R) = {sigma_a[-1]:.3e} ≠ 0"

    def test_solver_fields_nonnegative(self):
        """Все поля ≥ 0 в решении с включением."""
        from solver import solve_idr

        result = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        assert np.all(result["u"] >= -1e-10), f"|H|² < 0: min={result['u'].min():.3e}"
        assert np.all(result["v"] >= -1e-10), f"|E|² < 0: min={result['v'].min():.3e}"
        assert np.all(result["sigma_a"] >= -1e-10), f"σ_a < 0: min={result['sigma_a'].min():.3e}"

    def test_solver_grid_starts_at_r_inc(self):
        """Возвращаемая сетка r начинается с r_inc, а не с 0."""
        from solver import solve_idr

        r_inc = 0.004
        result = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=r_inc,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        r = result["r"]
        assert abs(r[0] - r_inc) < 1e-15, f"r[0]={r[0]}, ожидалось {r_inc}"
        assert abs(r[-1] - 0.012) < 1e-15, f"r[-1]={r[-1]}, ожидалось 0.012"

    def test_E_field_zero_at_inclusion_rises_inward(self):
        """
        Физика: E_φ(r_inc) = 0 и |E|² возрастает от включения к стенке,
        т.к. поле индуцируется в зазоре.
        """
        from solver import solve_idr

        result = solve_idr(
            N=100, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        v = result["v"]
        # v[0] = 0, максимум где-то внутри
        assert v[0] < v[len(v) // 2] or v[0] < v[-2], (
            "E² не возрастает от включения"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T5: Интеграционные / физические тесты
# ═══════════════════════════════════════════════════════════════════════════════

class TestInclusionPhysics:
    """Физическая корректность решения с включением."""

    def test_joule_power_positive(self):
        """Полная мощность джоулева нагрева P > 0."""
        from solver import solve_idr
        from postprocess import joule_dissipation, total_power

        result = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        r = result["r"]
        sigma_a = result["sigma_a"]
        sigma_p = result["sigma_p"]
        u = result["u"]
        v = result["v"]
        mod2 = sigma_a**2 + sigma_p**2

        Q = joule_dissipation(r, sigma_a, mod2, u, v)
        P = total_power(r, Q)

        assert P > 0, f"Мощность P = {P:.3e} ≤ 0"
        assert np.all(Q >= -1e-15), f"Q < 0: min={Q.min():.3e}"

    def test_sigma_maximum_inside_gap(self):
        """
        Максимум σ_a находится внутри кольцевого зазора [r_inc, R],
        а не на стенках (где σ=0).
        """
        from solver import solve_idr

        result = solve_idr(
            N=100, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        sigma_a = result["sigma_a"]
        i_max = np.argmax(sigma_a)

        assert 0 < i_max < len(sigma_a) - 1, (
            f"Максимум σ_a на стенке: i={i_max}, σ_a[i]={sigma_a[i_max]:.3e}"
        )

    def test_H_monotone_wall_to_inclusion(self):
        """
        |H|² убывает от стенки (r=R) к включению (r=r_inc):
        магнитное поле экранируется плазмой.
        """
        from solver import solve_idr

        result = solve_idr(
            N=100, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-5, relax=0.5,
        )

        u = result["u"]
        # u[-1] = H_wall² (максимум)
        assert u[-1] >= u[0], (
            f"|H|²(R)={u[-1]:.3e} < |H|²(r_inc)={u[0]:.3e}"
        )

    def test_grid_independence_annular(self):
        """
        Решения при N=80 и N=160 совпадают по σ_a с точностью ≤ 2%.
        """
        from solver import solve_idr

        res80 = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-6, relax=0.5,
        )
        res160 = solve_idr(
            N=160, R=0.012, p_pa=133.0,
            H_wall=100000.0,
            r_inc=0.004,
            max_iter=500, tol=1e-6, relax=0.5,
        )

        sigma80_interp = np.interp(res160["r"], res80["r"], res80["sigma_a"])
        scale = np.max(np.abs(res160["sigma_a"][:-1])) + 1e-30
        diff = np.max(np.abs(sigma80_interp[:-1] - res160["sigma_a"][:-1])) / scale

        assert diff < 0.02, (
            f"Сеточная ошибка N80 vs N160: {diff:.3%} > 2%"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T6: Совместимость — r_inc = 0 эквивалент без включения
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """r_inc=0 (дефолт) должен давать такие же результаты, как без включения."""

    def test_solver_r_inc_zero_equals_default(self):
        """
        solve_idr(r_inc=0.0) ≡ solve_idr() без параметра r_inc.
        """
        from solver import solve_idr

        result_default = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=1.0,
            max_iter=200, tol=1e-5, relax=0.5,
        )
        result_zero = solve_idr(
            N=80, R=0.012, p_pa=133.0,
            H_wall=1.0,
            r_inc=0.0,
            max_iter=200, tol=1e-5, relax=0.5,
        )

        assert np.allclose(result_default["u"], result_zero["u"], rtol=1e-8), (
            "u отличается при r_inc=0 и без r_inc"
        )
        assert np.allclose(result_default["v"], result_zero["v"], rtol=1e-8), (
            "v отличается при r_inc=0 и без r_inc"
        )
        assert np.allclose(result_default["sigma_a"], result_zero["sigma_a"], rtol=1e-8), (
            "sigma_a отличается при r_inc=0 и без r_inc"
        )

    def test_existing_tests_unaffected(self):
        """
        Решатель без параметра r_inc продолжает работать (обратная совместимость).
        Воспроизводим тест 5.1 — физические ограничения.
        """
        from solver import solve_idr

        H_wall = 1.0
        result = solve_idr(N=80, R=0.012, p_pa=133.0,
                           H_wall=H_wall, max_iter=300,
                           tol=1e-5, relax=0.5)

        u = result["u"]
        v = result["v"]
        sigma_a = result["sigma_a"]

        assert np.all(u >= -1e-10)
        assert np.all(v >= -1e-10)
        assert np.all(sigma_a >= -1e-10)
        assert abs(sigma_a[-1]) < 1e-8
        assert abs(u[-1] - H_wall**2) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
