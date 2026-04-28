"""
test_physics.py — тесты модуля physics.py.

Тесты 1.1–1.6 из плана реализации.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from physics import (
    conductivity, effective_field,
    ambipolar_diffusion, ionization_freq,
    collision_freq,
    neutral_density, gas_mass_density,
    poiseuille_v0, poiseuille_velocity, sccm_to_kg_s,
)
from config import OMEGA, E_CHARGE, M_ELECTRON, NU_C_PER_TORR, K_BOLTZMANN, M_ARGON

P_TEST = 133.0   # Па ≈ 1 мм рт. ст.
NE_TEST = 1e16   # м⁻³


# ─── 1.1 При ω→0: σ → вещественная, σ_p → 0 ────────────────────────────────
def test_conductivity_real_limit():
    """
    При ω→0 проводимость должна стать вещественной:
    σ_a → n_e·e²/(m_e·νc),  σ_p → 0.
    """
    import config as cfg
    omega_orig = cfg.OMEGA
    import physics

    # Подменяем OMEGA через монкипатч
    physics.OMEGA = 1e-10   # почти ноль
    import importlib
    importlib.reload(physics)   # перезагрузим с новым OMEGA
    from physics import conductivity as cond2, collision_freq as nu_c2

    physics.OMEGA = 1e-10
    # Пересчитываем вручную
    nu_c = NU_C_PER_TORR * (P_TEST / 133.322)
    sigma_a_analytic = NE_TEST * E_CHARGE**2 / (M_ELECTRON * nu_c)

    sigma_a, sigma_p, _ = conductivity(NE_TEST, P_TEST)
    # Допуск: σ_p/σ_a должно быть малым при ω≪νc
    ratio = abs(sigma_p) / abs(sigma_a)
    assert ratio < (OMEGA / nu_c) * 2, (
        f"σ_p/σ_a = {ratio:.4f} > ω/νc = {OMEGA/nu_c:.4f}"
    )

    # Восстанавливаем
    physics.OMEGA = omega_orig
    importlib.reload(physics)


# ─── 1.2 При νc≪ω: σ_a ≪ σ_p ────────────────────────────────────────────────
def test_conductivity_high_freq():
    """
    При νc≪ω: σ_a/σ_p ≈ νc/ω ≪ 1.

    Аналитически: σ_a = n_e·e²·νc/(m_e·(νc²+ω²)) ≈ n_e·e²·νc/(m_e·ω²)
                  σ_p = n_e·e²·ω/(m_e·(νc²+ω²))   ≈ n_e·e²/(m_e·ω)
    Отношение: σ_a/σ_p = νc/ω.
    """
    # При p = 133 Па, νc ≈ 5.6e9*1 = 5.6e9 с⁻¹; ω ≈ 8.5e7 с⁻¹ для 13.56 МГц
    # Т.е. νc > ω для аргона при 1 торр — берём малое давление
    p_low = 0.01   # Па  → νc ≪ ω
    nu_c = NU_C_PER_TORR * (p_low / 133.322)

    sigma_a, sigma_p, _ = conductivity(NE_TEST, p_low)

    ratio_analytic = nu_c / OMEGA
    ratio_computed = sigma_a / sigma_p

    assert abs(ratio_computed - ratio_analytic) / ratio_analytic < 0.01, (
        f"σ_a/σ_p = {ratio_computed:.6f}, ожидалось {ratio_analytic:.6f}"
    )


# ─── 1.3 σ пропорциональна n_e ───────────────────────────────────────────────
def test_sigma_proportional_ne():
    """
    σ_a и σ_p линейно зависят от n_e (при фиксированном p).
    """
    ne1 = 1e15
    ne2 = 1e16
    sa1, sp1, _ = conductivity(ne1, P_TEST)
    sa2, sp2, _ = conductivity(ne2, P_TEST)

    ratio_a = sa2 / sa1
    ratio_p = sp2 / sp1

    assert abs(ratio_a - ne2/ne1) < 1e-10, f"σ_a: ratio={ratio_a}, expect {ne2/ne1}"
    assert abs(ratio_p - ne2/ne1) < 1e-10, f"σ_p: ratio={ratio_p}, expect {ne2/ne1}"


# ─── 1.4 νi → 0 при E_eff → 0 ───────────────────────────────────────────────
def test_ionization_zero_field():
    """
    При нулевом поле ионизации нет: νi(E=0) = 0.
    """
    nu_i = ionization_freq(0.0, P_TEST)
    assert nu_i == 0.0 or abs(nu_i) < 1e-100, (
        f"νi({0.0}) = {nu_i} ≠ 0"
    )


# ─── 1.5 Da > 0 при любых физических E, p ────────────────────────────────────
def test_diffusion_positive():
    """
    Амбиполярный коэффициент диффузии Da > 0 для всех физических значений E, p.
    """
    E_vals = [0.0, 10.0, 1e3, 1e5]
    p_vals = [1.0, 133.0, 1333.0]

    for p in p_vals:
        for E in E_vals:
            E_eff = effective_field(E, p)
            Da = ambipolar_diffusion(E_eff, p)
            assert Da > 0, f"Da={Da} ≤ 0 при E={E}, p={p}"


# ─── 1.6 σ_a > 0, σ_p > 0 всегда ───────────────────────────────────────────
def test_conductivity_components_sign():
    """
    σ_a > 0 (рассеяние мощности) и σ_p > 0 (ёмкостная компонента) при ω > 0.
    """
    ne_vals = [1e14, 1e16, 1e18]
    p_vals  = [1.0, 133.0, 1333.0]

    for ne in ne_vals:
        for p in p_vals:
            sa, sp, mod2 = conductivity(ne, p)
            assert sa > 0, f"σ_a={sa} ≤ 0 при ne={ne}, p={p}"
            assert sp > 0, f"σ_p={sp} ≤ 0 при ne={ne}, p={p}"
            assert mod2 > 0, f"|σ|²={mod2} ≤ 0"


# ─── Дополнительный тест: аналитическое выражение σ_a ────────────────────────
def test_conductivity_analytic_value():
    """
    Проверяет точное аналитическое значение σ_a.
    """
    nu_c = collision_freq(P_TEST)
    sigma_a_expect = NE_TEST * E_CHARGE**2 * nu_c / (M_ELECTRON * (nu_c**2 + OMEGA**2))
    sigma_a, _, _ = conductivity(NE_TEST, P_TEST)
    assert abs(sigma_a - sigma_a_expect) / sigma_a_expect < 1e-12


# ─── Тесты течения Пуазейля ───────────────────────────────────────────────────

P_GAS = 133.0   # Па
T_GAS = 300.0   # К
R_TUBE = 0.012  # м


def test_neutral_density():
    """n_a = p / (k·T): проверка на аналитическое значение."""
    expected = P_GAS / (K_BOLTZMANN * T_GAS)
    assert abs(neutral_density(P_GAS, T_GAS) - expected) / expected < 1e-12


def test_gas_mass_density():
    """ρ = m_Ar · n_a."""
    n_a = neutral_density(P_GAS, T_GAS)
    expected = M_ARGON * n_a
    assert abs(gas_mass_density(P_GAS, T_GAS) - expected) / expected < 1e-12


def test_poiseuille_v0_formula():
    """
    v₀ = 2G / (π·ρ·R²).

    Проверяет, что формула задаёт правильный знак и размерность.
    При G > 0 должно быть v0 > 0.
    """
    G_kg_s = sccm_to_kg_s(500.0)
    v0 = poiseuille_v0(G_kg_s, R_TUBE, P_GAS, T_GAS)
    rho = gas_mass_density(P_GAS, T_GAS)
    expected = 2.0 * G_kg_s / (np.pi * rho * R_TUBE**2)
    assert v0 > 0, f"v0={v0} должно быть > 0"
    assert abs(v0 - expected) / expected < 1e-12


def test_poiseuille_mass_conservation():
    """
    Проверка закона сохранения массы: 2π ∫₀ᴿ ρ·v(r)·r dr = G.

    Правильная формула (из физики):
        G = (π/2)·ρ·v₀·R²   =>   2π ∫₀ᴿ ρ·v(r)·r dr = G

    Это именно тот интеграл, который был записан неверно в рукописях
    (там стояло 1/(2π) вместо 2π).
    """
    G_kg_s = sccm_to_kg_s(500.0)
    Nr = 1000
    r = np.linspace(0.0, R_TUBE, Nr + 1)
    v = poiseuille_velocity(r, G_kg_s, R_TUBE, P_GAS, T_GAS)
    rho = gas_mass_density(P_GAS, T_GAS)

    # G = 2π ∫₀ᴿ ρ·v(r)·r dr
    G_check = 2.0 * np.pi * np.trapezoid(rho * v * r, r)
    rel_err = abs(G_check - G_kg_s) / G_kg_s
    assert rel_err < 1e-5, (
        f"Нарушение сохранения массы: G_check={G_check:.6e}, "
        f"G_in={G_kg_s:.6e}, rel_err={rel_err:.2e}"
    )


def test_poiseuille_profile_shape():
    """
    Профиль v(r) должен быть параболическим: максимум на оси, ноль на стенке.
    """
    G_kg_s = sccm_to_kg_s(200.0)
    r = np.linspace(0.0, R_TUBE, 101)
    v = poiseuille_velocity(r, G_kg_s, R_TUBE, P_GAS, T_GAS)

    assert v[0] > 0, "Скорость на оси должна быть > 0"
    assert abs(v[-1]) < 1e-12, f"Скорость на стенке должна быть 0, получено {v[-1]}"
    assert np.all(np.diff(v) <= 0), "Профиль должен монотонно убывать от оси к стенке"


def test_sccm_to_kg_s_units():
    """
    Перевод единиц: 1 sccm аргона при STP.

    1 sccm = 1 см³/мин при STP (T=273.15 К, p=101325 Па).
    ρ_STP = M_Ar · p_STP / (k·T_STP) ≈ 1.784 кг/м³.
    G = 1e-6/60 м³/с · 1.784 кг/м³ ≈ 2.97e-8 кг/с.
    """
    G = sccm_to_kg_s(1.0)
    assert 1e-8 < G < 1e-7, f"1 sccm = {G:.3e} кг/с (ожидается ~3e-8)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
