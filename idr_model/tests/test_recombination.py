"""
test_recombination.py — тесты уравнения (13) с членом объёмной рекомбинации.

Уравнение без рекомбинации (текущее):
  Da·∇²ne + νi·ne = 0

Уравнение с рекомбинацией (новое):
  Da·∇²ne + νi·ne − β·ne² = 0

При высоких давлениях (1000–10000 Па) и ne ~ 10²² м⁻³ рекомбинация
доминирует над диффузией, и без этого члена модель неприменима.

Тестируемый модуль: equations.py (после добавления β_recomb параметра
в build_sigma_equation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from equations import make_grid, build_sigma_equation


# ── Вспомогательные ─────────────────────────────────────────────────────────

def _uniform_arrays(N, value):
    """Создаёт массив длины N+1 с одинаковым значением."""
    return np.full(N + 1, value)


# ── Обратная совместимость ───────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Без β_recomb поведение должно быть идентичным текущему."""

    def test_no_recombination_by_default(self):
        """build_sigma_equation без β_recomb = текущему поведению."""
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 100.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0)

        # Вызов без beta_recomb (по умолчанию = 0 или None)
        l, m, u, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma_ref)

        # Должен быть power iteration: rhs = νi · σ_ref
        # Проверяем на внутреннем узле
        i = N // 2
        expected_rhs = nu_i[i] * sigma_ref[i]
        assert rhs[i] == pytest.approx(expected_rhs, rel=1e-10), \
            f"rhs[{i}] = {rhs[i]:.6e}, ожидалось {expected_rhs:.6e}"


# ── Тесты с рекомбинацией ───────────────────────────────────────────────────

class TestRecombinationTerm:
    """Проверка, что член −β·σ² корректно добавляется в уравнение."""

    def test_recombination_modifies_rhs(self):
        """
        При β > 0 правая часть должна содержать −β·σ_ref²:
          (-Da·Δ)·σ_new = νi·σ_ref − β·σ_ref²
        """
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 1000.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 100.0  # Большой σ
        beta_recomb = 1e-13  # м³/с (типичное для Ar)

        l, m, u, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma_ref,
                                              beta_recomb=beta_recomb)

        # На внутреннем узле: rhs = νi·σ_ref − β·σ_ref²
        i = N // 2
        expected = nu_i[i] * sigma_ref[i] - beta_recomb * sigma_ref[i] ** 2
        assert rhs[i] == pytest.approx(expected, rel=1e-10), \
            f"rhs[{i}] = {rhs[i]:.6e}, ожидалось {expected:.6e}"

    def test_recombination_reduces_rhs(self):
        """rhs с рекомбинацией < rhs без рекомбинации (при σ > 0)."""
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 500.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 50.0

        l0, m0, u0, rhs0 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref)
        l1, m1, u1, rhs1 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref,
                                                   beta_recomb=1e-12)

        # Внутренние узлы (кроме границ): rhs1 < rhs0
        inner = slice(1, N)
        assert np.all(rhs1[inner] <= rhs0[inner] + 1e-30), \
            "Рекомбинация не уменьшила rhs"

    def test_zero_beta_is_no_op(self):
        """β = 0 эквивалентно отсутствию рекомбинации."""
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 200.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 10.0

        l0, m0, u0, rhs0 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref)
        l1, m1, u1, rhs1 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref,
                                                   beta_recomb=0.0)

        np.testing.assert_allclose(rhs0, rhs1, rtol=1e-14)
        np.testing.assert_allclose(m0, m1, rtol=1e-14)

    def test_boundary_conditions_preserved(self):
        """Рекомбинация не влияет на граничные условия σ(R) = 0."""
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 500.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 50.0

        l, m, u, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma_ref,
                                              beta_recomb=1e-12)

        # ГУ на стенке: σ(R) = 0 → main[N]=1, rhs[N]=0
        assert m[N] == pytest.approx(1.0)
        assert rhs[N] == pytest.approx(0.0)

        # ГУ на оси: dσ/dr = 0 (симметрия)
        assert l[0] == pytest.approx(0.0)


# ── Физические предсказания ──────────────────────────────────────────────────

class TestRecombinationPhysics:
    """
    Качественные физические проверки решения с рекомбинацией.
    """

    def test_equilibrium_ne_estimate(self):
        """
        При доминировании рекомбинации (β·ne >> Da·∇²ne):
          νi·ne ≈ β·ne²  →  ne_eq ≈ νi/β

        Это локальный баланс — ne определяется не профилем, а локальным E.
        """
        nu_i = 1e4     # с⁻¹, типичная частота ионизации
        beta = 1e-13   # м³/с, коэфф. рекомбинации Ar

        ne_eq = nu_i / beta
        assert ne_eq == pytest.approx(1e17, rel=0.01), \
            f"ne_eq = {ne_eq:.2e}, ожидалось ~1e17"

    def test_high_pressure_ne_bounded(self):
        """
        При β > 0, решение ne не может уйти в бесконечность:
        член −β·ne² ограничивает рост.

        Проверяем: ne_max из решения уравнения (13) с β
        меньше, чем ne_max без β (при тех же νi и Da).
        """
        from solver import thomas_solve

        N = 50
        R = 0.001   # 1 мм — масштаб джета
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-5)     # м²/с (высокое давление)
        nu_i = _uniform_arrays(N, 1e4)    # с⁻¹
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 1e6

        # Без рекомбинации
        l0, m0, u0, rhs0 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref)
        sol0 = thomas_solve(l0, m0, u0, rhs0)

        # С рекомбинацией
        l1, m1, u1, rhs1 = build_sigma_equation(r, h, Da, nu_i,
                                                   sigma_a_ref=sigma_ref,
                                                   beta_recomb=1e-13)
        sol1 = thomas_solve(l1, m1, u1, rhs1)

        assert sol1.max() <= sol0.max() + 1e-10, \
            f"Решение с β ({sol1.max():.3e}) > без β ({sol0.max():.3e})"


# ── Тесты с IMEX-методом (dt) ───────────────────────────────────────────────

class TestRecombinationIMEX:
    """Рекомбинация корректно работает совместно с псевдо-временным шагом."""

    def test_imex_with_recombination(self):
        """build_sigma_equation с dt + beta_recomb не падает."""
        N = 20
        R = 0.01
        r, h = make_grid(N, R)
        Da = _uniform_arrays(N, 1e-3)
        nu_i = _uniform_arrays(N, 500.0)
        sigma_ref = np.maximum(1.0 - (r / R) ** 2, 0.0) * 10.0

        l, m, u, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma_ref,
                                              dt=1e-5,
                                              beta_recomb=1e-13)

        # Проверка: rhs содержит все три слагаемых (dt + νi - β)
        i = N // 2
        inv_dt = 1.0 / 1e-5
        expected = sigma_ref[i] * (inv_dt + nu_i[i]) - 1e-13 * sigma_ref[i] ** 2
        assert rhs[i] == pytest.approx(expected, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
