"""
test_verification_2d.py — верификация 2D решателя на известных решениях
и сходимости сетки (по методологии из конспекта).

Четыре проверки:
  1. Форма профиля σ(r,z) ~ J₀(j₀₁·r/R)·sin(π·z/L)
  2. Предел большого L: λ₀²(Dirichlet) → j₀₁²·D/ν при L→∞
  3. Аннулярная геометрия: λ₀² совпадает с первым нулём Бесселя в кольце
  4. Сходимость O(h²) при измельчении сетки

Всё работает только с compute_lambda0_2d (без Maxwell-решателя),
поэтому тесты быстрые.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.special import j0, y0
from scipy.optimize import brentq
from equations_2d import make_grid_2d


# ===========================================================================
# 1. Форма профиля σ(r,z)
# ===========================================================================

class TestProfileShape:

    def test_sigma_matches_J0_sin(self):
        """
        При однородных Da=D, νi=ν, Дирихле на всех границах точный
        собственный вектор:

            σ(r, z) = J₀(j₀₁·r/R) · sin(π·z/L)

        Проверяем: нормированная корреляция вычисленного профиля
        с аналитическим > 0.99 (цилиндрический L²-скаляр произведение).

        Допуск: при Nr=100, Nz=50 дискретизационная погрешность < 1%.
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 1.0, 1.0
        Nr, Nz = 100, 50
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)

        D, nu = 1.0, 1.0
        Da   = np.full((Nr + 1, Nz + 1), D)
        nu_i = np.full((Nr + 1, Nz + 1), nu)

        _, sigma = compute_lambda0_2d(
            r, z, hr, hz, Da, nu_i,
            bc_z_sigma="dirichlet", return_profile=True,
        )

        # Аналитический профиль
        j01 = 2.4048255577
        r_col  = r[:, np.newaxis]
        z_row  = z[np.newaxis, :]
        sigma_exact = j0(j01 * r_col / R) * np.sin(np.pi * z_row / L)
        sigma_exact[-1, :] = 0.0
        sigma_exact[:, 0]  = 0.0
        sigma_exact[:, -1] = 0.0

        # Нормируем по цилиндрическому скалярному произведению ∫∫·r dr dz
        def cyl_ip(f, g):
            return float(np.trapezoid(
                np.trapezoid(f * g * r_col, z, axis=1), r))

        norm_c  = np.sqrt(cyl_ip(sigma, sigma))
        norm_e  = np.sqrt(cyl_ip(sigma_exact, sigma_exact))
        cos_sim = abs(cyl_ip(sigma / norm_c, sigma_exact / norm_e))

        assert cos_sim > 0.99, (
            f"Косинусное сходство профиля = {cos_sim:.4f} < 0.99\n"
            f"Профиль не соответствует J0*sin"
        )


# ===========================================================================
# 2. Предел большого L: Дирихле → 1D
# ===========================================================================

class TestLargeLLimit:

    def test_dirichlet_approaches_1d_as_L_grows(self):
        """
        При Дирихле на торцах: λ₀² = D·[(j₀₁/R)² + (π/L)²]/ν

        При L→∞ осевой вклад (π/L)²→0 и λ₀²→D·(j₀₁/R)²/ν (1D-предел).

        Проверяем:
          - Монотонное убывание λ₀² с ростом L
          - Последнее значение (L=20·R) отличается от 1D-предела < 3%
        """
        from self_consistent_2d import compute_lambda0_2d

        R   = 1.0
        Nr  = 80
        D, nu = 1.0, 1.0
        j01 = 2.4048255577
        lam_1d = D * (j01 / R)**2 / nu   # предел при L→∞

        L_values = [1.0, 2.0, 5.0, 10.0, 20.0]
        lam_values = []

        for L in L_values:
            Nz = max(10, int(Nr * L / R / 2))   # пропорционально L
            r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
            Da   = np.full((Nr + 1, Nz + 1), D)
            nu_i = np.full((Nr + 1, Nz + 1), nu)
            lam = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="dirichlet")
            lam_values.append(lam)

        # Монотонное убывание
        for i in range(len(lam_values) - 1):
            assert lam_values[i] > lam_values[i + 1], (
                f"λ₀² не убывает: L={L_values[i]:.0f} → {lam_values[i]:.4f}, "
                f"L={L_values[i+1]:.0f} → {lam_values[i+1]:.4f}"
            )

        # Сходимость к 1D-пределу
        rel_err = abs(lam_values[-1] - lam_1d) / lam_1d
        assert rel_err < 0.03, (
            f"При L=20: λ₀²={lam_values[-1]:.4f}, "
            f"1D-предел={lam_1d:.4f}, погрешность={rel_err:.2%}"
        )


# ===========================================================================
# 3. Аннулярная геометрия: первый ноль кросс-произведения Бесселя
# ===========================================================================

class TestAnnularBessel:

    def test_annular_lambda0_matches_bessel_zero(self):
        """
        Для кольца [r_inc, R] с Da=D, νi=ν, Neumann по z (чисто радиальная
        задача). Уравнение:

            (1/r)d/dr(r·dσ/dr) + k²·σ = 0,  σ(r_inc)=σ(R)=0

        Решение: A·J₀(k·r) + B·Y₀(k·r). Ненулевое при:

            J₀(k·R)·Y₀(k·r_inc) - J₀(k·r_inc)·Y₀(k·R) = 0

        В коде λ₀² определяется как λ₀² = D·k₁² / ν (см. compute_lambda0):
        power iteration на (-D·Δ)σ_new = ν·σ_old находит μ = ν/(D·k₁²),
        затем λ₀² = 1/μ = D·k₁²/ν.
        При λ₀=1: D·k₁²/ν = 1 → k₁ = √(ν/D) — условие Шоттки.

        Допуск 2% (дискретизация по r).
        """
        from self_consistent_2d import compute_lambda0_2d

        R     = 1.0
        r_inc = 0.3
        Nr, Nz = 120, 10
        D, nu = 1.0, 1.0

        r, z, hr, hz = make_grid_2d(Nr, Nz, R, R * 0.5, r_min=r_inc)

        Da   = np.full((Nr + 1, Nz + 1), D)
        nu_i = np.full((Nr + 1, Nz + 1), nu)

        lam0_sq_num = compute_lambda0_2d(
            r, z, hr, hz, Da, nu_i, bc_z_sigma="neumann")

        # Аналитический первый ноль кросс-произведения Бесселя.
        # Скобку находим сканированием: ищем первую смену знака.
        def cross_bessel(mu):
            return j0(mu * R) * y0(mu * r_inc) - j0(mu * r_inc) * y0(mu * R)

        mu_scan = np.linspace(0.5, 20.0, 2000)
        vals = np.array([cross_bessel(m) for m in mu_scan])
        idx = np.where(vals[:-1] * vals[1:] < 0)[0]
        assert len(idx) > 0, "Первый ноль кросс-Бесселя не найден"
        mu1 = brentq(cross_bessel, mu_scan[idx[0]], mu_scan[idx[0] + 1])
        lam0_sq_exact = D * mu1**2 / nu   # λ₀² = D·k₁²/ν (соглашение кода)

        rel_err = abs(lam0_sq_num - lam0_sq_exact) / lam0_sq_exact
        assert rel_err < 0.02, (
            f"λ₀²_num={lam0_sq_num:.4f}, "
            f"λ₀²_exact={lam0_sq_exact:.4f} (mu1={mu1:.4f}), "
            f"погрешность={rel_err:.2%}"
        )


# ===========================================================================
# 4. Сходимость O(h²) при измельчении сетки
# ===========================================================================

class TestGridConvergence:

    def test_lambda0_second_order_convergence(self):
        """
        Метод конечных разностей 2-го порядка: при удвоении сетки
        ошибка уменьшается в ~4 раза.

        Точное значение (Da=D, νi=ν, R=1, L=1, Дирихле):
            λ₀²_exact = D·[(j₀₁/R)² + (π/L)²] / ν
                      = (2.405)² + π² ≈ 15.653

        Сетки: Nr = 20, 40, 80 (Nz пропорционально).
        Ожидаемый коэффициент сходимости: 3.5–4.5 (учитываем оба направления).
        """
        from self_consistent_2d import compute_lambda0_2d

        R, L = 1.0, 1.0
        D, nu = 1.0, 1.0
        j01 = 2.4048255577
        lam_exact = D * ((j01 / R)**2 + (np.pi / L)**2) / nu

        grids = [(20, 10), (40, 20), (80, 40)]
        errors = []

        for Nr, Nz in grids:
            r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
            Da   = np.full((Nr + 1, Nz + 1), D)
            nu_i = np.full((Nr + 1, Nz + 1), nu)
            lam = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                     bc_z_sigma="dirichlet")
            errors.append(abs(lam - lam_exact))

        # Коэффициент сходимости при удвоении сетки
        ratio_1 = errors[0] / errors[1]   # Nr: 20 → 40
        ratio_2 = errors[1] / errors[2]   # Nr: 40 → 80

        assert ratio_1 > 3.5, (
            f"Коэффициент сходимости (20→40) = {ratio_1:.2f} < 3.5\n"
            f"Ошибки: {errors[0]:.2e}, {errors[1]:.2e}, {errors[2]:.2e}"
        )
        assert ratio_2 > 3.5, (
            f"Коэффициент сходимости (40→80) = {ratio_2:.2f} < 3.5\n"
            f"Ошибки: {errors[0]:.2e}, {errors[1]:.2e}, {errors[2]:.2e}"
        )

        if __name__ == "__main__":
            for (Nr, Nz), err in zip(grids, errors):
                print(f"  Nr={Nr:3d}, Nz={Nz:2d}: err={err:.2e}")
            print(f"  Коэффициенты: {ratio_1:.2f}, {ratio_2:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
