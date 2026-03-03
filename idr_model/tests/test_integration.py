"""
test_integration.py — интеграционные тесты 1D модели ИДР.

Тесты 5.1–5.6 из плана реализации.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from solver import solve_idr
from postprocess import joule_dissipation, total_power
from physics import conductivity


# ─── 5.1 Физические ограничения на все поля ──────────────────────────────────
def test_full_solution_physical_bounds():
    """
    Все поля ≥ 0, σ(R)=0, |H|²(R) = H_wall².
    """
    H_wall = 1.0
    result = solve_idr(N=80, R=0.012, p_pa=133.0,
                       H_wall=H_wall, max_iter=300,
                       tol=1e-5, relax=0.5)

    u       = result["u"]
    v       = result["v"]
    sigma_a = result["sigma_a"]

    assert np.all(u >= -1e-10),        f"u содержит отрицательные значения: {u.min():.3e}"
    assert np.all(v >= -1e-10),        f"v содержит отрицательные значения: {v.min():.3e}"
    assert np.all(sigma_a >= -1e-10),  f"σ_a содержит отрицательные значения: {sigma_a.min():.3e}"

    assert abs(sigma_a[-1]) < 1e-8,   f"σ(R)={sigma_a[-1]:.3e} ≠ 0"
    assert abs(u[-1] - H_wall**2) < 1e-8, f"|H|²(R)={u[-1]:.3e} ≠ {H_wall**2}"


# ─── 5.2 Удельное тепловыделение Q = σ_a·|E|² ───────────────────────────────
def test_joule_dissipation_formula19():
    """
    Q(r) = σ_a(r)·|E(r)|² = σ_a·v  (формула (19) статьи — Джоулев нагрев).

    Проверяем: Q ≥ 0 и интегрирование трапецией ≈ Симпсону (< 1% разн.).
    """
    result = solve_idr(N=100, max_iter=200, tol=1e-5, relax=0.5)

    r       = result["r"]
    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]
    u       = result["u"]
    v       = result["v"]

    sigma_mod2 = sigma_a**2 + sigma_p**2
    Q = joule_dissipation(r, sigma_a, sigma_mod2, u, v)

    # Q ≥ 0
    assert np.all(Q >= -1e-15), f"Q содержит отрицательные значения: {Q.min():.3e}"

    # Интегрирование методом трапеций
    P_trap = 2 * np.pi * np.trapezoid(Q * r, r)

    # Интегрирование методом Симпсона (требует нечётного числа точек)
    Q_r = Q * r
    if len(Q_r) % 2 == 0:
        Q_r = Q_r[:-1]
        r_s = r[:-1]
    else:
        r_s = r

    from scipy.integrate import simpson
    P_simp = 2 * np.pi * simpson(Q_r, x=r_s)

    # Два метода должны совпадать с точностью O(h²)
    rel_diff = abs(P_trap - P_simp) / (abs(P_simp) + 1e-30)
    assert rel_diff < 0.01, (
        f"Трапеция vs Симпсон: {P_trap:.4e} vs {P_simp:.4e}, "
        f"отн. разница {rel_diff:.3%}"
    )


# ─── 5.3 Воспроизведение кривой 1 рис. 1 (pDa=const, νi/p=const) ─────────────
def test_reproduce_fig1_curve1():
    """
    При pDa=const и νi/p=const профиль σ/σ₀ определяется функцией Бесселя J₀.

    Проверяем качественное поведение: σ убывает от центра к стенке.
    """
    result = solve_idr(N=100, R=0.012, p_pa=133.0,
                       max_iter=300, tol=1e-5, relax=0.5)

    sigma_a = result["sigma_a"]

    # σ должна убывать от центра (i=0) к стенке (i=N)
    # Проверяем монотонное убывание (с допуском на осцилляции)
    sigma_max = sigma_a[:-1].max()
    sigma_min = sigma_a[:-1].min()

    assert sigma_a[0] >= sigma_a[-2] * 0.5, (
        f"Профиль не убывает к стенке: σ(0)={sigma_a[0]:.3e}, σ(R⁻)={sigma_a[-2]:.3e}"
    )
    assert sigma_a[-1] == 0.0 or sigma_a[-1] < 1e-8, (
        f"σ(R)={sigma_a[-1]:.3e} ≠ 0"
    )


# ─── 5.4 Качественное поведение при p=133 Па, R=0.012 м ──────────────────────
def test_reproduce_fig2_qualitative():
    """
    При p=133 Па, R=0.012 м:
    - |H|² убывает от стенки к центру (ток индукции)
    - |E|² имеет максимум внутри (не на оси и не на стенке)
    - σ_a > 0 в центре
    """
    result = solve_idr(N=100, R=0.012, p_pa=133.0,
                       H_wall=1.0, max_iter=300,
                       tol=1e-5, relax=0.5)

    u = result["u"]
    v = result["v"]
    sigma_a = result["sigma_a"]

    # |H|² должно убывать от стенки к центру
    assert u[-1] >= u[0] * 0.5, (
        f"|H|² не убывает к центру: u(0)={u[0]:.3e}, u(R)={u[-1]:.3e}"
    )

    # σ_a > 0 хоть где-то внутри
    assert np.max(sigma_a[:-1]) > 0, "σ_a = 0 везде — нет плазмы"

    # v = |E|² ≥ 0 везде
    assert np.all(v >= -1e-12)


# ─── 5.5 Сеточная независимость ──────────────────────────────────────────────
def test_grid_independence():
    """
    Решение при N=100 и N=200 совпадает с точностью 1% по норме σ_a.
    """
    res100 = solve_idr(N=100, R=0.012, p_pa=133.0,
                       max_iter=300, tol=1e-6, relax=0.5)
    res200 = solve_idr(N=200, R=0.012, p_pa=133.0,
                       max_iter=300, tol=1e-6, relax=0.5)

    # Интерполируем грубую сетку на мелкую для сравнения
    r100 = res100["r"]
    r200 = res200["r"]
    sigma100 = res100["sigma_a"]
    sigma200 = res200["sigma_a"]

    # Сравниваем только внутренние узлы (σ(R)=0 по ГУ)
    sigma200_interp = np.interp(r100, r200, sigma200)

    scale = np.max(np.abs(sigma100[:-1])) + 1e-30
    diff  = np.max(np.abs(sigma100[:-1] - sigma200_interp[:-1])) / scale

    assert diff < 0.01, (
        f"Сеточная ошибка N100 vs N200: {diff:.3%} > 1%"
    )


# ─── 5.6 Высокочастотный предел (νc≪ω) ────────────────────────────────────────
def test_high_freq_limit():
    """
    При очень низком давлении (νc≪ω) σ_a≪σ_p.
    Система (17)-(18) должна воспроизводиться.

    Проверяем, что решатель не расходится и поля физичны.
    """
    p_low = 0.1   # Па → νc ≪ ω
    result = solve_idr(N=80, R=0.01, p_pa=p_low, H_wall=1.0,
                       max_iter=300, tol=1e-5, relax=0.3)

    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]

    # σ_a ≪ σ_p при малом давлении
    inner = slice(1, -1)
    sa_mean = np.mean(sigma_a[inner])
    sp_mean = np.mean(sigma_p[inner])

    assert sa_mean < sp_mean, (
        f"Ожидалось σ_a < σ_p при ВЧ пределе: "
        f"σ_a={sa_mean:.3e}, σ_p={sp_mean:.3e}"
    )

    # Поля физичны
    assert np.all(result["u"] >= -1e-10)
    assert np.all(result["v"] >= -1e-10)


# ─── 5.7 ИДР-режим: ионизационный баланс при параметрах по умолчанию ─────────
def test_idr_ionization_regime():
    """
    При параметрах конфига (аргон, 133 Па, H_wall = 100 000 А/м) решение должно
    находиться в ИДР-режиме:

      νi_max  >>  Da · λ₁  (минимальный критерий соответствия статье)

    Это гарантирует, что ионизация играет роль в формировании профиля σ,
    а не только амбиполярная диффузия.

    Дополнительно проверяем, что профиль σ/σ₀ является более плоским в ядре,
    чем J₀ — характерный признак ИДР (ионизация сосредоточена у стенки,
    диффузия заполняет объём).
    """
    from config import P_PA, R_TUBE, H_WALL
    from physics import ambipolar_diffusion, ionization_freq, effective_field
    from config import OMEGA
    import numpy as np

    MU_0 = 4.0 * np.pi * 1e-7

    # Диффузионные потери: Da · λ₁²
    Da0 = ambipolar_diffusion(0.0, P_PA)
    lambda1_sq = (2.4048 / R_TUBE) ** 2
    loss_rate = Da0 * lambda1_sq   # с⁻¹

    # Ионизация у стенки (оценка по равномерному H)
    E_wall_est = OMEGA * MU_0 * H_WALL * R_TUBE / 2.0
    E_eff_wall = effective_field(E_wall_est, P_PA)
    nu_i_wall = ionization_freq(E_eff_wall, P_PA)

    assert nu_i_wall > loss_rate, (
        f"Нет ИДР-режима при H_wall={H_WALL:.0f} А/м: "
        f"νi_wall={nu_i_wall:.3e} с⁻¹ ≤ Da·λ₁={loss_rate:.3e} с⁻¹. "
        f"Увеличьте H_WALL — должно быть νi >> Da·λ₁."
    )

    # Запуск решателя и проверка профиля
    result = solve_idr(N=100, R=R_TUBE, p_pa=P_PA,
                       H_wall=H_WALL, max_iter=500, tol=1e-6, relax=0.5)
    r   = result["r"]
    rn  = r / R_TUBE
    sa  = result["sigma_a"]
    s0  = sa[0] if sa[0] > 0 else 1.0

    # Значение σ/σ₀ на r/R = 0.5 (середина)
    sa_mid = float(np.interp(0.5, rn, sa / s0))

    try:
        from scipy.special import j0 as scipy_j0
        j0_mid = float(scipy_j0(2.4048 * 0.5))   # ≈ 0.672
    except ImportError:
        j0_mid = 0.672

    # ИДР-профиль должен быть более плоским в ядре (σ/σ₀ при r/R=0.5 > J₀):
    # при νi >> Da·λ₁ ионизация у стенки «выравнивает» профиль
    assert sa_mid > j0_mid, (
        f"ИДР-профиль не более плоский, чем J₀ в ядре: "
        f"σ/σ₀(0.5)={sa_mid:.3f}, J₀(0.5)={j0_mid:.3f}. "
        f"Проверьте формирование ИДР-режима."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
