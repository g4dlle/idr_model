"""
physics.py — физические аппроксимации для 1D модели ИДР.

Реализует формулы (20), (21) из статьи Абдуллина–Желтухина (1985) для аргона,
а также выражения для комплексной проводимости и эффективного поля.

Соглашения по единицам
─────────────────────
• Давление p передаётся в Па; внутри функций переводится в мм рт. ст. (торр)
  через PA_TO_TORR = 1/133.322.
• E_eff передаётся в В/м.
• Результаты Da — м²/с; νi — с⁻¹; σ — А/(В·м) = См/м.
"""

import numpy as np
from config import (
    E_CHARGE, M_ELECTRON, OMEGA,
    D1, D2, B1, B2, B3,
    NU_C_PER_TORR,
)

# Перевод единиц давления
PA_TO_TORR = 1.0 / 133.322   # 1 Па = 7.5006e-3 мм рт. ст.


def pressure_torr(p_pa: float | np.ndarray) -> float | np.ndarray:
    """Перевод давления из Па в мм рт. ст. (торр)."""
    return p_pa * PA_TO_TORR


def collision_freq(p_pa: float | np.ndarray) -> float | np.ndarray:
    """
    Частота упругих столкновений электронов с нейтралами (аргон).

    νc = NU_C_PER_TORR * p [торр]   [с⁻¹]

    Parameters
    ----------
    p_pa : давление в Па

    Returns
    -------
    νc в с⁻¹
    """
    return NU_C_PER_TORR * pressure_torr(p_pa)


def effective_field(E_abs: float | np.ndarray,
                    p_pa: float | np.ndarray) -> float | np.ndarray:
    """
    Эффективное электрическое поле (Es в обозначениях статьи, стр. 107).

    Аппроксимации (20) и (21) для Da и νi принимают именно это поле как аргумент.
    E_abs — амплитуда поля (пиковое значение).
    Действующее (RMS) значение синусоидального поля: E_rms = E_abs / sqrt(2).
    Учёт частоты (снижение нагрева при ω > νc):

        E_eff = (E_abs / sqrt(2)) * νc / sqrt(νc² + ω²)

    В коллизионном режиме (νc >> ω): E_eff ≈ E_abs / sqrt(2) = E_rms.
    В реактивном режиме (ω >> νc):   E_eff → 0 (электроны не успевают реагировать).

    Parameters
    ----------
    E_abs : |E| в В/м (амплитудное значение, = sqrt(|E|²))
    p_pa  : давление в Па

    Returns
    -------
    E_eff в В/м
    """
    nu_c = collision_freq(p_pa)
    return E_abs / np.sqrt(2.0) * nu_c / np.sqrt(nu_c**2 + OMEGA**2)


def ambipolar_diffusion(E_eff: float | np.ndarray,
                        p_pa: float | np.ndarray) -> float | np.ndarray:
    """
    Амбиполярный коэффициент диффузии (аппроксимация (20)).

    p·Da = D1 * exp(D2 * (E_eff/p)²),   [D1] = м²/с·торр
    => Da = D1/p * exp(D2*(E_eff/p)²)   [м²/с]

    где E_eff/p приведено к единицам В/(м·торр).

    Parameters
    ----------
    E_eff : эффективное поле в В/м
    p_pa  : давление в Па

    Returns
    -------
    Da в м²/с (всегда > 0)
    """
    p_torr = pressure_torr(p_pa)
    # Нормировка поля: В/(м·торр)
    Ep = E_eff / (p_torr + 1e-30)   # защита от p→0
    p_Da = D1 * np.exp(np.minimum(D2 * Ep**2, 700.0))  # clamp before overflow
    return p_Da / (p_torr + 1e-30)


def ionization_freq(E_eff: float | np.ndarray,
                    p_pa: float | np.ndarray) -> float | np.ndarray:
    """
    Частота ионизации (аппроксимация (21)).

    νi/p = [B1*(E_eff/p)^0.5 + B2*(E_eff/p)^1.5] * exp(-B3/(E_eff/p))

    => νi = p * (...)   [с⁻¹]

    При E_eff→0 функция стремится к 0 (нет ионизации).

    Parameters
    ----------
    E_eff : эффективное поле в В/м
    p_pa  : давление в Па

    Returns
    -------
    νi в с⁻¹ (≥ 0)
    """
    p_torr = pressure_torr(p_pa)
    Ep = E_eff / (p_torr + 1e-30)   # В/(м·торр)

    # Защита: при Ep→0 аргумент экспоненты → -∞
    safe_Ep = np.where(Ep > 1e-30, Ep, 1e-30) if isinstance(Ep, np.ndarray) \
              else max(Ep, 1e-30)

    poly = B1 * safe_Ep**0.5 + B2 * safe_Ep**1.5
    exponent = np.exp(-B3 / safe_Ep)

    nu_i_over_p = poly * exponent
    # Если E_eff = 0, принудительно обнуляем
    zero_mask = (E_eff == 0.0)
    if isinstance(nu_i_over_p, np.ndarray):
        nu_i_over_p = np.where(zero_mask, 0.0, nu_i_over_p)
    elif E_eff == 0.0:
        nu_i_over_p = 0.0

    return p_torr * nu_i_over_p


def conductivity(n_e: float | np.ndarray,
                 p_pa: float | np.ndarray
                 ) -> tuple:
    """
    Комплексная электронная проводимость.

    σ = n_e·e² / (m_e · (νc - i·ω))
      = n_e·e²·(νc + i·ω) / (m_e · (νc² + ω²))

    Возвращает (σ_a, σ_p, sigma_mod2) где:
        σ_a  — активная (вещественная) часть [См/м]
        σ_p  — реактивная (мнимая)  часть [См/м]
        sigma_mod2 = σ_a² + σ_p²  [См²/м²]

    Parameters
    ----------
    n_e  : концентрация электронов [м⁻³]
    p_pa : давление в Па

    Notes
    -----
    Знак: при ω > 0 мнимая часть σ_p > 0 (ток опережает поле).
    """
    nu_c = collision_freq(p_pa)
    denom = nu_c**2 + OMEGA**2
    prefactor = n_e * E_CHARGE**2 / M_ELECTRON

    sigma_a = prefactor * nu_c / denom
    sigma_p = prefactor * OMEGA / denom
    sigma_mod2 = sigma_a**2 + sigma_p**2

    return sigma_a, sigma_p, sigma_mod2


def sigma_from_conductivity(sigma_a: np.ndarray,
                             sigma_p: np.ndarray) -> tuple:
    """
    Вычисляет |σ|² и полезные соотношения из компонент проводимости.

    Returns (|σ|², σ_a/|σ|²)
    """
    sigma_mod2 = sigma_a**2 + sigma_p**2
    # Защита от деления на ноль
    safe_mod2 = np.where(sigma_mod2 > 0, sigma_mod2, 1e-300)
    return sigma_mod2, sigma_a / safe_mod2
