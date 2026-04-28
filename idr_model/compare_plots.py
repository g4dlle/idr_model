"""
compare_plots.py — графики сравнения модели с экспериментальными данными ВЧИ.

Запуск:
    uv run python idr_model/compare_plots.py
    uv run python idr_model/compare_plots.py --save  # сохранить в plots/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import j0 as J0, iv as Iv
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(__file__))
from physics import (
    collision_freq, effective_field, ionization_freq,
    ambipolar_diffusion, conductivity,
)
from self_consistent import solve_maxwell_for_ne0
import config as cfg

# ── Настройки графиков ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

mu0   = 4 * np.pi * 1e-7
omega = cfg.OMEGA
R     = 0.012   # м, радиус трубки
lam1  = 2.405   # первый ноль J0

# ════════════════════════════════════════════════════════════════════════════
# Экспериментальные данные (оцифровано из графиков)
# ════════════════════════════════════════════════════════════════════════════

# Рис. 2.27, кривая 3 — ВЧИ, аргон, G=0: H_z(r) [10² А/м]
exp_227_r  = np.array([0.19, 4.74, 8.64, 11.81])          # мм
exp_227_Hz = np.array([41.50, 41.64, 43.93, 54.59])        # ×10² А/м

# Рис. 2.33 — ВЧИ, аргон, G=0, кривая 3: j_φ(r) [10⁶ А/м²]
exp_233_r  = np.array([0.05, 2.46, 5.13, 8.02, 11.17])    # мм
exp_233_j  = np.array([0.00, 0.39, 1.51, 1.82, 1.83])     # ×10⁶ А/м²

# Рис. 2.41 — n_e vs P_p при f=1.76 МГц, аргон (кривая 1, G=0.05 г/с)
exp_241_P  = np.array([0.73, 0.98, 1.17, 1.34])            # кВт
exp_241_ne = np.array([1.97, 6.93, 19.07, 53.37]) * 1e15   # м⁻³

# Рис. 2.38 — n_e vs p при P_p=1.5 кВт, аргон, кривая 4
exp_238_p  = np.array([27.94, 46.74, 118.91, 160.85, 221.84, 262.87])  # Па
exp_238_ne = np.array([1.37, 1.55, 3.21, 3.71, 1.98, 1.16]) * 1e16    # м⁻³

# Рис. 2.34 — j_φ(z) — ВЧИ, аргон, G=0 (кривая 1)
exp_234_z  = np.array([-176.87, -119.64, -61.10, 32.55])   # мм
exp_234_j  = np.array([0.69, 1.88, 0.38, 0.0])             # ×10⁶ А/м²


# ════════════════════════════════════════════════════════════════════════════
# Модельные вычисления
# ════════════════════════════════════════════════════════════════════════════

def skin_depth(n_e, p_pa):
    sa, _, _ = conductivity(n_e, p_pa)
    return (2.0 / (omega * mu0 * sa)) ** 0.5 if sa > 0 else np.inf


def H_profile_numerical(n_e0, p_pa, H_wall, N=80):
    """Нормированный профиль H(r)/H(R) из численного решателя."""
    res = solve_maxwell_for_ne0(n_e0=n_e0, N=N, R=R, p_pa=p_pa,
                                H_wall=H_wall, max_iter=500, tol=1e-5)
    r_grid = res["r"]
    H_abs  = np.sqrt(np.maximum(res["u"], 0.0))
    H_norm = H_abs / H_abs[-1] if H_abs[-1] > 0 else H_abs
    return r_grid, H_norm, res["lambda0"]


def j_profile_numerical(n_e0, p_pa, H_wall, N=80, delta_sheath=0.0):
    """
    Численный профиль |j_φ(r)| = σ_a(r) · |E_φ(r)|.
    E_φ вычисляется из решателя (v = |E_φ|²).

    Parameters
    ----------
    delta_sheath : ширина оболочки [м]; 0 — стандартное ГУ n_e(R)=0.
                   При delta_sheath > 0 плазма ограничена r ≤ R−delta_sheath,
                   а в оболочке [R−delta_sheath, R] n_e = 0.
    """
    res   = solve_maxwell_for_ne0(n_e0=n_e0, N=N, R=R, p_pa=p_pa,
                                  H_wall=H_wall, max_iter=500, tol=1e-5,
                                  delta_sheath=delta_sheath)
    r_grid = res["r"]
    E      = np.sqrt(np.maximum(res["v"], 0.0))
    j      = res["sigma_a"] * E
    # нормируем на глобальный максимум j (может быть в оболочке, где E велико,
    # но sigma_a=0 — там j=0; реальный максимум всегда внутри плазмы)
    jmax = np.max(j)
    j_norm = j / jmax if jmax > 0 else j
    return r_grid, j_norm


def bessel_H_profile(x_ratio: float, N: int = 300) -> tuple:
    """
    Аналитический профиль H_z(r)/H_z(R) для однородной цилиндрической плазмы.

    Решение уравнения Максвелла в однородной плазме:
        H_z(r) = I₀(k·r) / I₀(k·R),   k = (1+i)/delta

    Параметры
    ----------
    x_ratio : R/delta — отношение радиуса трубки к скин-глубине.
    N       : число точек сетки.

    Возвращает (r_mm, H_norm), где H_norm[0] = H(0)/H(R), H_norm[-1] = 1.
    """
    rr = np.linspace(0.0, R, N)
    k  = (1.0 + 1j) * x_ratio / R       # k = (1+i)/delta = (1+i)*x_ratio/R
    H  = np.abs(Iv(0, k * rr)) / abs(Iv(0, k * R))
    return rr * 1e3, H                   # r в мм


def fit_bessel_ratio(target_ratio: float = 0.760) -> float:
    """
    Находит x = R/delta, при котором H(0)/H(R) = 1/|I₀((1+i)x)| = target_ratio.

    При x→0: H(0)/H(R) → 1;  при x→∞: H(0)/H(R) → 0.
    """
    def eq(x):
        return 1.0 / abs(Iv(0, (1.0 + 1j) * x)) - target_ratio
    return brentq(eq, 0.01, 20.0)


def implied_ne_from_delta(delta_m: float, p_pa: float) -> float:
    """
    Концентрация n_e, соответствующая заданной скин-глубине delta (в м).

        sigma_a = 2 / (omega * mu0 * delta²)
        n_e = sigma_a * m_e * (nu_c² + omega²) / (e² * nu_c)
    """
    sigma_a = 2.0 / (omega * mu0 * delta_m**2)
    nu_c = collision_freq(p_pa)
    ne = sigma_a * 9.1093837015e-31 * (nu_c**2 + omega**2) / (1.602176634e-19**2 * nu_c)
    return ne


def find_E_eff_particle_balance(p_pa: float) -> float:
    """
    Находит E_eff [В/м] из уравнения баланса частиц:

        ν_i(E_eff, p) = D_a(E_eff, p) × (λ₁/R)²

    Это поле, которое плазма устанавливает при данном давлении независимо
    от мощности (определяется только геометрией и газом).

    Returns
    -------
    E_eff в В/м, или np.nan если решение не найдено.
    """
    def balance(E_abs):
        Ef = effective_field(E_abs, p_pa)
        return ionization_freq(Ef, p_pa) - ambipolar_diffusion(Ef, p_pa) * (lam1 / R)**2

    # Сканирование для поиска скобки знакопеременности
    E_scan = np.logspace(2.0, 6.5, 800)   # 100 В/м … 3 МВ/м
    vals   = np.array([balance(E) for E in E_scan])
    idx    = np.where(vals[:-1] * vals[1:] < 0)[0]
    if len(idx) == 0:
        return np.nan
    return brentq(balance, E_scan[idx[0]], E_scan[idx[0] + 1], xtol=1.0)


def n_e_two_balance(p_pa: float, P_pp: float = 1500.0,
                    V_eff: float | None = None) -> float:
    """
    Концентрация электронов из двух балансов при фиксированной мощности P_pp [Вт].

    1. Баланс частиц → E_eff(p)
    2. Баланс мощности:
           P_pp = σ_a · E_abs² · V_eff
           E_abs = E_eff · √2 · √(νc²+ω²) / νc
           σ_a = n_e·e²·νc / (m_e·(νc²+ω²))
       →   n_e = P_pp · m_e · νc / (2·e² · E_eff² · V_eff)

    При δ >> R (нет скин-эффекта на эксп. уровне n_e) V_eff = π·R²·L_TUBE.

    Returns
    -------
    n_e в м⁻³, или np.nan если баланс частиц не решается.
    """
    if V_eff is None:
        import config as _cfg
        V_eff = np.pi * R**2 * _cfg.L_TUBE   # π·R²·L

    E_eff = find_E_eff_particle_balance(p_pa)
    if np.isnan(E_eff):
        return np.nan

    nu_c = collision_freq(p_pa)
    e    = 1.602176634e-19
    m_e  = 9.1093837015e-31
    return P_pp * m_e * nu_c / (2.0 * e**2 * E_eff**2 * V_eff)


def threshold_Hwall(p_pa):
    """H_wall, при котором ν_i = D_a·λ₁²/R² (порог разряда)."""
    def balance(H):
        E_w   = omega * mu0 * H * R / 2
        E_eff = effective_field(E_w, p_pa)
        nu_i  = ionization_freq(E_eff, p_pa)
        Da    = ambipolar_diffusion(E_eff, p_pa)
        return nu_i - Da * lam1**2 / R**2
    return brentq(balance, 1e3, 1e6)


def nu_ratio(H_wall, p_pa):
    """ν_i / (D_a·λ₁²/R²) — кратность ионизации."""
    E_w   = omega * mu0 * H_wall * R / 2
    E_eff = effective_field(E_w, p_pa)
    nu_i  = ionization_freq(E_eff, p_pa)
    Da    = ambipolar_diffusion(E_eff, p_pa)
    loss  = Da * lam1**2 / R**2
    return nu_i / loss if loss > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Построение
# ════════════════════════════════════════════════════════════════════════════

def make_figure():
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "Сравнение модели ВЧИ с экспериментальными данными\n"
        r"Аргон, $f = 1{,}76$ МГц, $R = 12$ мм",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.07, right=0.97,
                           top=0.91, bottom=0.08,
                           hspace=0.40, wspace=0.38)

    # ── График 1: Профиль H_z(r) ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    p_pa   = 133.0

    # Нормировка эксперимента на стеночное значение
    Hz_norm_exp = exp_227_Hz / exp_227_Hz[-1]

    # ── (а) Аналитический Bessel-фит: H = I₀((1+i)r/δ) / I₀((1+i)R/δ) ──
    # Находим R/δ, при котором H(0)/H(R) = 0.760 (по эксп. данным)
    target_ratio = exp_227_Hz[0] / exp_227_Hz[-1]   # = 0.760
    x_fit   = fit_bessel_ratio(target_ratio)          # R/delta
    delta_m = R / x_fit                              # скин-глубина, м
    ne_impl = implied_ne_from_delta(delta_m, p_pa)   # подразумеваемая n_e

    r_bes, H_bes = bessel_H_profile(x_fit)
    ax1.plot(r_bes, H_bes, color="#d62728", lw=2.2, ls="-",
             label=(rf"Bessel-фит ($\delta/R={1/x_fit:.2f}$,"
                    + "\n" + rf"$n_e\approx{ne_impl:.1e}$ м$^{{-3}}$)"))

    # ── (б) Численный профиль при n_e, близком к фиту ─────────────────────
    ne_num = 5.5e20   # даёт H(0)/H(R) ≈ 0.76 численно
    r_num, H_num, lam0_num = H_profile_numerical(ne_num, p_pa, cfg.H_WALL, N=100)
    ax1.plot(r_num * 1e3, H_num, color="#ff7f0e", lw=1.8, ls="--",
             label=(rf"Числ. ($n_{{e0}}={ne_num:.1e}$ м$^{{-3}}$,"
                    + "\n" + rf"$\lambda_0={lam0_num:.2f}$, $\delta/R={R/x_fit:.2f}$)"))

    # ── (в) Справочный плоский профиль при экспериментальной плотности ────
    ne_exp_meas = 3.21e16   # эксп. n_e из рис. 2.38, p≈119 Па
    nu_c_val = collision_freq(p_pa)
    sa_exp = (ne_exp_meas * 1.602176634e-19**2 * nu_c_val /
              (9.1093837015e-31 * (nu_c_val**2 + omega**2)))
    delta_exp = np.sqrt(2 / (omega * mu0 * sa_exp))
    r_flat, H_flat, _ = H_profile_numerical(ne_exp_meas, p_pa, cfg.H_WALL, N=80)
    ax1.plot(r_flat * 1e3, H_flat, color="#1f77b4", lw=1.4, ls=":",
             label=(rf"Числ. ($n_{{e0}}={ne_exp_meas:.1e}$,"
                    + rf" $\delta/R={delta_exp/R:.0f}$) — плоский"))

    # ── Экспериментальные точки ────────────────────────────────────────────
    ax1.plot(exp_227_r, Hz_norm_exp, "ks", ms=8, zorder=5,
             label="Эксп. Рис. 2.27 (аргон, $G=0$)")

    ax1.set_xlabel("$r$, мм")
    ax1.set_ylabel(r"$H_z(r)\,/\,H_z(R)$")
    ax1.set_title(r"Профиль $H_z(r)$" + "\n(Рис. 2.27, кривая 3)")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0.68, 1.04)
    ax1.legend(fontsize=7.5, loc="lower right")

    # ── График 2: Профиль j_φ(r) ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    # Экспериментальные точки (нормированы на максимум)
    j_exp_norm = exp_233_j / np.max(exp_233_j)

    # Численный профиль: j = sigma_a * |E_phi| из решателя
    # Три варианта граничного условия для n_e (ширина оболочки):
    #   delta=0   → n_e(R)=0 (Дирихле), j(R)=0
    #   delta=1мм → плазма ограничена R_eff=11мм, оболочка 1мм
    #   delta=2мм → плазма ограничена R_eff=10мм, оболочка 2мм
    ne0_exp = 3.71e17   # экспериментальная концентрация

    r_j0, j_mod0 = j_profile_numerical(ne0_exp, p_pa, cfg.H_WALL,
                                        delta_sheath=0.0)
    r_j1, j_mod1 = j_profile_numerical(ne0_exp, p_pa, cfg.H_WALL,
                                        delta_sheath=1e-3)
    r_j2, j_mod2 = j_profile_numerical(ne0_exp, p_pa, cfg.H_WALL,
                                        delta_sheath=2e-3)

    ax2.plot(r_j0 * 1e3, j_mod0, "#1f77b4", lw=1.8,
             label=r"Модель: $\delta_{sh}=0$ (ГУ $n_e(R){=}0$)")
    ax2.plot(r_j1 * 1e3, j_mod1, "#ff7f0e", lw=1.8, ls="--",
             label=r"Модель: $\delta_{sh}=1$ мм ($R_{eff}{=}11$ мм)")
    ax2.plot(r_j2 * 1e3, j_mod2, "#2ca02c", lw=1.8, ls=":",
             label=r"Модель: $\delta_{sh}=2$ мм ($R_{eff}{=}10$ мм)")
    ax2.plot(exp_233_r, j_exp_norm, "ks", ms=7, zorder=5,
             label="Эксп. Рис. 2.33\n(ВЧИ, аргон, $G=0$)")
    ax2.set_xlabel("$r$, мм")
    ax2.set_ylabel(r"$j_\varphi(r)\,/\,j_{\varphi,\max}$")
    ax2.set_title(r"Профиль $j_\varphi(r)$ = $\sigma_a\cdot|E_\varphi|$"
                  + "\n(Рис. 2.33, кривая 3)")
    ax2.set_xlim(0, 12)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=8, loc="upper left")

    # ── График 3: n_e vs P — сверхлинейность (Рис. 2.41) ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])

    # Эксперимент
    ax3.semilogy(exp_241_P, exp_241_ne, "ks", ms=7, label="Эксп. (Рис. 2.41)")
    # Степенная аппроксимация эксперимента
    log_P  = np.log(exp_241_P)
    log_ne = np.log(exp_241_ne)
    alpha  = np.polyfit(log_P, log_ne, 1)
    P_fit  = np.linspace(0.70, 1.40, 100)
    ne_fit = np.exp(np.polyval(alpha, np.log(P_fit)))
    ax3.semilogy(P_fit, ne_fit, "k--", lw=1.4,
                 label=rf"Аппрокс.: $n_e \propto P^{{{alpha[0]:.1f}}}$")

    # Модель: ν_i/потери → n_e_eff при разных H_wall
    # n_e ∝ (ν_i/потери)^β, нормируем на экспериментальную точку
    H_arr    = np.linspace(4.5e4, 7.5e4, 60)
    ratio_arr= np.array([nu_ratio(H, 133.0) for H in H_arr])
    # Степень: из ln(n_e)/ln(ν_i/loss) ~ const
    ratio_exp = np.array([nu_ratio(H, 133.0)
                          for H in np.linspace(4.9e4, 6.5e4, 4)])
    # Нормировка: модель на n_e при P=0.98 кВт (среднее)
    ne_ref   = exp_241_ne[1]
    rat_ref  = nu_ratio(5.3e4, 133.0)
    # Power = C * H^2 => P_model ~ H^2
    P_model  = (H_arr / H_arr[0])**2 * exp_241_P[0]
    # ne_model ~ ratio^β, подобрать β из данных
    beta_arr = np.log(ratio_arr / ratio_arr[0] + 1e-30)
    ne_model = ne_ref * ratio_arr / rat_ref
    ax3.semilogy(P_model[ratio_arr > 0.5], ne_model[ratio_arr > 0.5],
                 "#1f77b4", lw=1.8, label=r"Модель: $n_e \propto \nu_i/\Gamma_{diff}$")

    ax3.set_xlabel("$P_p$, кВт")
    ax3.set_ylabel("$n_e$, м⁻³")
    ax3.set_title("$n_e$ vs мощность разряда\n(Рис. 2.41, $f=1{,}76$ МГц)")
    ax3.set_xlim(0.6, 1.6)
    ax3.legend(fontsize=8)

    # ── График 4: n_e vs p — немонотонность (Рис. 2.38) ──────────────────
    ax4 = fig.add_subplot(gs[1, 0])

    # ── Двухбалансная модель — показываем оба конкурирующих механизма ────
    #
    # n_e ∝ νc(p) / E_abs²(p)  где E_abs(p) — из баланса частиц
    #
    # Конкурирующие механизмы:
    #   νc ∝ p                   → n_e растёт с p (меньше потери через диффузию)
    #   E_abs(p) ∝ p·x(p)       → растёт с p → n_e убывает (ионизация неэффективна)
    # Пик там, где d/dp[νc/E_abs²] = 0.
    # Модель даёт пик при p≈6 Па (экcп. пик при ~160 Па).
    #
    p_wide = np.logspace(np.log10(1.5), np.log10(310), 200)
    E_abs_arr = np.array([find_E_eff_particle_balance(pp) for pp in p_wide])
    valid_w   = np.isfinite(E_abs_arr) & (E_abs_arr > 0)
    nu_c_arr  = np.array([collision_freq(pp) for pp in p_wide])

    # Два компонента (нормированы к значению при p=100 Па)
    p_ref_idx = np.argmin(np.abs(p_wide - 100))
    fac_nc  = nu_c_arr / nu_c_arr[p_ref_idx]               # νc/νc_ref
    fac_Eab = E_abs_arr**2 / E_abs_arr[p_ref_idx]**2       # E_abs²/E_abs_ref²
    ne_2b_w = fac_nc / fac_Eab                             # n_e ∝ νc/E_abs²

    # Нормируем обе (эксперимент и модель) к своему максимуму
    ne_exp_norm   = exp_238_ne / np.max(exp_238_ne)
    ne_2b_norm    = ne_2b_w / np.nanmax(ne_2b_w[valid_w])

    # Экспериментальные точки (нормированные)
    ax4.plot(exp_238_p, ne_exp_norm, "rs", ms=8, zorder=5,
             label="Эксп. (Рис. 2.38, норм.)")

    # Кривая модели
    ax4.plot(p_wide[valid_w], ne_2b_norm[valid_w], "#1f77b4", lw=2.0,
             label=r"Модель: $n_e \propto \nu_c\,/\,E_{abs}^2$"
                   + "\n(два баланса, норм.)")

    # Два компонента раздельно (не нормированные на свой макс, а просто отмасштабированные)
    ax4.plot(p_wide[valid_w], fac_nc[valid_w] / np.max(fac_nc[valid_w]),
             "#2ca02c", lw=1.2, ls="--", alpha=0.75,
             label=r"Фактор $\nu_c \propto p$ (↑)")
    ax4.plot(p_wide[valid_w],
             1.0 / (fac_Eab[valid_w] / np.nanmax(fac_Eab[valid_w])),
             "#d62728", lw=1.2, ls=":", alpha=0.75,
             label=r"Фактор $1/E_{abs}^2$ (↓ при росте $p$)")

    # Пик модели и пик эксперимента
    p_peak_mod = p_wide[valid_w][np.argmax(ne_2b_norm[valid_w])]
    p_peak_exp = exp_238_p[np.argmax(exp_238_ne)]
    ax4.axvline(p_peak_mod, color="#1f77b4", ls="--", lw=1.2, alpha=0.8)
    ax4.axvline(p_peak_exp, color="r",       ls="--", lw=1.2, alpha=0.8)
    ax4.text(p_peak_mod + 2, 0.65,
             rf"$p_{{opt}}^{{mod}}$≈{p_peak_mod:.0f} Па",
             color="#1f77b4", fontsize=7.5, va="center")
    ax4.text(p_peak_exp + 2, 0.50,
             rf"$p_{{opt}}^{{exp}}$≈{p_peak_exp:.0f} Па",
             color="r", fontsize=7.5, va="center")

    ax4.set_xlabel("$p$, Па")
    ax4.set_ylabel(r"$n_e\,/\,n_{e,\max}$ (нормировано)")
    ax4.set_title("$n_e$ vs давление\n(Рис. 2.38, $P_p=1{,}5$ кВт)")
    ax4.set_xscale("log")
    ax4.set_xlim(1.5, 320)
    ax4.set_ylim(-0.05, 1.15)
    ax4.legend(fontsize=7.0, loc="lower right")

    # ── График 5: Порог разряда ν_i/потери vs H_wall ─────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    H_scan = np.linspace(2e4, 1.1e5, 300)
    colors_p  = ["#2ca02c", "#1f77b4", "#d62728"]
    press_arr = [66.5, 133.0, 266.0]
    labels_p  = ["$p=66{,}5$ Па", "$p=133$ Па", "$p=266$ Па"]

    for pp, col, lab in zip(press_arr, colors_p, labels_p):
        ratio_scan = np.array([nu_ratio(H, pp) for H in H_scan])
        ax5.semilogy(H_scan / 1e3, np.clip(ratio_scan, 1e-6, None),
                     color=col, lw=1.8, label=lab)
        # Порог
        try:
            H_thr = threshold_Hwall(pp)
            ax5.axvline(H_thr / 1e3, color=col, ls="--", lw=1.0, alpha=0.7)
        except Exception:
            pass

    ax5.axhline(1.0, color="k", ls="-", lw=1.2, label="Порог ($\\nu_i = \\Gamma$)")
    ax5.set_xlabel("$H_\mathrm{wall}$, кА/м")
    ax5.set_ylabel(r"$\nu_i\,/\,(D_a\lambda_1^2/R^2)$")
    ax5.set_title("Порог самоподдержания разряда\n"
                  r"$\nu_i = D_a\lambda_1^2/R^2$")
    ax5.set_xlim(H_scan[0] / 1e3, H_scan[-1] / 1e3)
    ax5.set_ylim(1e-4, 1e4)
    ax5.legend(fontsize=8, loc="upper left")

    # Аннотации порогов
    for pp, col, lab in zip(press_arr, colors_p, labels_p):
        try:
            H_thr = threshold_Hwall(pp)
            ax5.text(H_thr / 1e3 + 1, 5e-4,
                     f"{H_thr/1e3:.0f} кА/м",
                     color=col, fontsize=8, rotation=90, va="bottom")
        except Exception:
            pass

    # ── График 6: Скин-глубина δ vs n_e ──────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ne_arr = np.logspace(15, 20, 200)

    colors_pp = ["#2ca02c", "#1f77b4", "#d62728"]
    for pp, col, lab in zip(press_arr, colors_pp, labels_p):
        delta_arr = np.array([skin_depth(ne, pp) for ne in ne_arr])
        ax6.loglog(ne_arr, delta_arr * 100, color=col, lw=1.8, label=lab)

    # Линия δ = R
    ax6.axhline(R * 100, color="k", ls="--", lw=1.2, label=f"$\\delta = R={R*100:.1f}$ см")

    # Экспериментальные точки: n_e из Рис. 2.38 (p=133 Па)
    ne_exp_pts = np.array([0.22, 0.81, 1.75, 3.71, 7.83, 22.4]) * 1e16
    delta_exp  = np.array([skin_depth(ne, 133.0) for ne in ne_exp_pts])
    ax6.loglog(ne_exp_pts, delta_exp * 100, "ks", ms=6,
               label="Эксп. точки\n(Рис. 2.38, $p=133$ Па)")

    ax6.set_xlabel("$n_e$, м⁻³")
    ax6.set_ylabel("$\\delta$, см")
    ax6.set_title("Глубина скин-слоя $\\delta(n_e)$\n"
                  "при экспериментальных условиях")
    ax6.set_xlim(1e15, 1e20)
    ax6.legend(fontsize=8, loc="upper right")

    return fig


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", nargs="?", const="plots", metavar="DIR",
                        help="Сохранить PNG в указанную папку (по умолчанию plots/)")
    args = parser.parse_args()

    fig = make_figure()

    if args.save is not None:
        out_dir = args.save
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "comparison_model_vs_exp.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
