"""
compare_trends.py — влияние давления, мощности и расхода на параметры разряда.

Три сравнительных панели:
  1. H_z(0)/H_z(R) vs давление p      — Рис. 2.30 (кривые 1 и 4, Ar, z=−60 мм)
  2. j_φ(r≈R) vs мощность P_p          — Рис. 2.35 (кривая 1, Ar, z=−120 мм, G=0)
  3. j_φ(r≈R) vs расход G              — Рис. 2.36 (кривая 1, Ar, z=−120 мм)

Методология:
  • Панель 1: n_e0 фиксируется бисекцией при p=133 Па → H(0)/H(R) = exp. значение.
    Затем p варьируется при том же n_e0.  Это раскрывает зависимость скин-эффекта
    от давления при постоянной концентрации.
  • Панель 2: варьируется H_wall (P ∝ ∫σ_a|E|²dV), остальные параметры постоянны.
    Кривые нормируются на референсную точку (p=133 Па, H_wall=150 кА/м).
  • Панель 3: модель 1D не содержит расхода (Pe≪1), поэтому j = const.
    Показывается предсказание модели (горизонтальная линия) vs эксп. scatter.

Запуск:
    uv run python idr_model/compare_trends.py
    uv run python idr_model/compare_trends.py --save   # сохранить в plots/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from self_consistent import solve_maxwell_for_ne0
import config as cfg
from physics import SCCM_TO_KG_S

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

R = 0.012    # м, радиус трубки
L = 0.12     # м, длина индуктора (для вычисления полной мощности)

# ════════════════════════════════════════════════════════════════════════════
# Экспериментальные данные (оцифровано из графиков)
# ════════════════════════════════════════════════════════════════════════════

# --- Рис. 2.30: H_z vs давление (аргон, z=−60 мм) --------------------------
# Кривая 1 (r=R, Ar): H_z(R)
_p230_R = np.array([36.07, 99.20, 133.47, 234.47, 298.50, 441.88])   # Па
_H230_R = np.array([28.77, 28.21, 27.74, 28.77, 29.25, 27.92]) * 1e2  # А/м

# Кривая 4 (r=0, Ar): H_z(0)
_p230_0 = np.array([36.97, 99.20, 135.27, 234.47, 280.46, 442.79])   # Па
_H230_0 = np.array([22.55, 22.83, 22.83, 22.55, 23.40, 22.83]) * 1e2  # А/м

# Паруем точки по ближайшему давлению
exp_230_p     = 0.5 * (_p230_R + _p230_0)          # среднее давление
exp_230_ratio = _H230_0 / _H230_R                   # H(0)/H(R)

# --- Рис. 2.35: j_φ vs мощность (Ar, z=−120 мм, G=0) -----------------------
exp_235_P_kW = np.array([1.52, 2.00, 2.46, 2.64])    # кВт
exp_235_j    = np.array([1.21, 1.40, 1.65, 1.74]) * 1e6  # А/м²

# --- Рис. 2.36: j_φ vs расход (Ar, z=−120 мм) ------------------------------
exp_236_G_gs = np.array([0.01, 0.04, 0.10, 0.19])    # г/с
exp_236_j    = np.array([1.78, 1.25, 1.36, 1.14]) * 1e6  # А/м²

# Перевод г/с → sccm (аргон)
exp_236_G_sccm = exp_236_G_gs * 1e-3 / SCCM_TO_KG_S  # ≈ 336, 1343, 3358, 6379

# ════════════════════════════════════════════════════════════════════════════
# Параметры модельных расчётов
# ════════════════════════════════════════════════════════════════════════════

P_PA_REF  = 133.0       # Па, базовое давление
H_WALL_REF = cfg.H_WALL  # А/м, базовое поле (150 кА/м)

# n_e0 для воспроизведения скин-эффекта (панель 1 и 3).
# Бисекция даёт n_e0 ≈ 6–7×10²⁰ для H(0)/H(R) = 0.80 при p=133 Па.
# Оценка: δ = sqrt(2/(ωμ₀σ_a)) = R/1.5 = 8 мм → σ_a ≈ 2000 С/м → n_e ≈ 4×10²⁰.
NE_H_INIT_LO = 3e20    # нижн.: H(0)/H(R) ≈ 0.93 > target → нужно больше n_e
NE_H_INIT_HI = 9e20    # верхн.: H(0)/H(R) ≈ 0.68 < target

# Целевое H(0)/H(R) из Рис. 2.30 при p ≈ 133 Па
TARGET_RATIO_230 = float(np.mean(exp_230_ratio))   # ≈ 0.80

# n_e0 для расчёта профиля j (панель 2 и 3); заимствовано из compare_plots.py
NE_J = 3.71e17   # м⁻³

# ════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════════════════════════════════

def _solve(ne0, p_pa=P_PA_REF, h_wall=H_WALL_REF, N=60, tol=1e-4):
    """
    Запуск solve_maxwell_for_ne0 с адаптивными параметрами.

    При больших n_e или малых давлениях (сильный скин-эффект, δ << R)
    требуется малый коэффициент под-релаксации.  При неудаче повторно
    запускается с более консервативными параметрами.
    """
    from physics import collision_freq, conductivity
    import numpy as np

    # Оцениваем σ_a для выбора параметров: σ_a ≈ n_e·e²/(m_e·νc)
    nu_c = collision_freq(p_pa)
    from config import E_CHARGE, M_ELECTRON, OMEGA
    denom = nu_c**2 + OMEGA**2
    sigma_est = ne0 * E_CHARGE**2 / M_ELECTRON * nu_c / denom

    if sigma_est >= 5000:       # δ < 5 мм, очень сильный скин-эффект
        relax, max_iter = 0.08, 2000
    elif sigma_est >= 2000:     # δ ~ 8–10 мм
        relax, max_iter = 0.12, 1200
    elif sigma_est >= 500:
        relax, max_iter = 0.20, 800
    elif sigma_est >= 100:
        relax, max_iter = 0.30, 500
    else:
        relax, max_iter = 0.50, 300

    res = solve_maxwell_for_ne0(
        n_e0=ne0, N=N, R=R, p_pa=p_pa,
        H_wall=h_wall, max_iter=max_iter, tol=tol, relax=relax,
    )
    # Повторная попытка с меньшей релаксацией и большим числом итераций
    if not res["converged"] and relax > 0.05:
        res2 = solve_maxwell_for_ne0(
            n_e0=ne0, N=N, R=R, p_pa=p_pa,
            H_wall=h_wall, max_iter=max_iter * 2, tol=tol, relax=relax / 2,
        )
        if res2["converged"]:
            return res2
    return res


def _h_ratio(res):
    """H(0)/H(R) из результата solve_maxwell_for_ne0."""
    u = res["u"]
    Hr = np.sqrt(max(u[-1], 0.0))
    H0 = np.sqrt(max(u[0],  0.0))
    return H0 / Hr if Hr > 0 else 1.0


def _j_peak(res):
    """Максимальная азимутальная плотность тока j_φ = σ_a · E_φ."""
    j = res["sigma_a"] * np.sqrt(np.maximum(res["v"], 0.0))
    return float(np.max(j))


def _power_per_length(res):
    """Мощность, поглощаемая плазмой, Вт/м (интеграл π·∫σ_a·|E|²·r·dr)."""
    sigma_a = res["sigma_a"]
    v       = res["v"]
    r       = res["r"]
    integrand = sigma_a * v * r
    return float(np.pi * np.trapezoid(integrand, r))


def bisect_ne_for_h_ratio(target, p_pa=P_PA_REF, h_wall=H_WALL_REF,
                           n_iter=12, lo=NE_H_INIT_LO, hi=NE_H_INIT_HI):
    """Бисекция по n_e0 → H(0)/H(R) = target (1D, log-пространство)."""
    for i in range(n_iter):
        ne_mid = np.sqrt(lo * hi)
        res    = _solve(ne_mid, p_pa=p_pa, h_wall=h_wall)
        ratio  = _h_ratio(res)
        print(f"  bisect [{i+1:2d}/{n_iter}]: n_e={ne_mid:.3e}  "
              f"H(0)/H(R)={ratio:.4f}  target={target:.4f}  "
              f"conv={res['converged']}")
        if ratio > target:
            lo = ne_mid
        else:
            hi = ne_mid
    return float(np.sqrt(lo * hi))


# ════════════════════════════════════════════════════════════════════════════
# Вычисление зависимостей для трёх панелей
# ════════════════════════════════════════════════════════════════════════════

def panel1_h_ratio_vs_pressure(ne0_ref):
    """
    H(0)/H(R) vs давление при фиксированном n_e0.

    Физика: рост p увеличивает νc → уменьшает σ_a (∝ 1/νc при νc≫ω)
    → увеличивает δ → ослабляет скин-эффект → H(0)/H(R) растёт с p.
    Экспериментально H(0)/H(R) почти не меняется с p — это означает,
    что в реальном разряде n_e также изменяется, компенсируя эффект давления.
    """
    pressures = np.array([36, 66, 100, 133, 200, 300, 400, 442], dtype=float)
    ratios    = np.zeros_like(pressures)
    for k, p in enumerate(pressures):
        # При малых давлениях скин-эффект очень сильный (σ_a ~ 1/p большое):
        # требуется малый relax и много итераций → передаём ne0 как есть,
        # а _solve сам выберет параметры по ne0.
        res = _solve(ne0_ref, p_pa=p)
        sigma_max = float(np.max(res["sigma_a"]))
        if res["converged"]:
            ratios[k] = _h_ratio(res)
        else:
            ratios[k] = np.nan   # модель неустойчива при очень сильном скин-эффекте
        print(f"  p={p:6.1f} Pa  H(0)/H(R)={ratios[k]:.4f}  "
              f"sigma_max={sigma_max:.1f} S/m  conv={res['converged']}")
    return pressures, ratios


def panel2_j_vs_power(ne0_ref):
    """
    j_φ(peak) и P_p vs H_wall при фиксированном n_e0.

    H_wall — прокси мощности: P ∝ ∫σ_a|E|²dV ∝ H_wall² (приближённо).
    При слабом скин-эффекте (δ≫R): E_φ ≈ ωμ₀H_wall·R/2 → j ∝ H_wall → P ∝ H_wall²
    → j ∝ P^0.5.  При сильном скин-эффекте показатель возрастает (шкин-слой
    насыщается — n_e или σ_a меняются нелинейно).

    Используем n_e0 = NE_J (из compare_plots.py) для нормальной амплитуды j.
    """
    h_walls = np.array([70, 90, 110, 130, 150, 175, 200, 230]) * 1e3  # А/м
    j_vals  = np.zeros(len(h_walls))
    P_vals  = np.zeros(len(h_walls))
    for k, hw in enumerate(h_walls):
        res     = _solve(ne0_ref, h_wall=hw)
        j_vals[k] = _j_peak(res)
        P_vals[k] = _power_per_length(res) * L    # Вт/м × м = Вт
        print(f"  H_wall={hw/1e3:.0f} kA/m  "
              f"j_peak={j_vals[k]:.3e} A/m2  "
              f"P={P_vals[k]/1e3:.3f} kW  conv={res['converged']}")
    return h_walls, j_vals, P_vals


def panel3_j_vs_flow(ne0_ref):
    """
    j_φ(peak) при изменении расхода (1D модель).

    Поскольку 1D модель не содержит осевого течения, j_φ не зависит от G.
    Модель воспроизводит предел Pe≪1: диффузионный перенос на порядки
    превышает конвективный → G не влияет на радиальный профиль.

    Возвращает скалярное значение j (предсказание модели) для отображения
    в виде горизонтальной линии на фоне экспериментальных точек.
    """
    res = _solve(ne0_ref, p_pa=P_PA_REF, h_wall=H_WALL_REF)
    j_model = _j_peak(res)
    print(f"  j_model (1D, G-independent) = {j_model:.3e} A/m2  conv={res['converged']}")
    return j_model


# ════════════════════════════════════════════════════════════════════════════
# Построение фигуры
# ════════════════════════════════════════════════════════════════════════════

def make_figure():
    # ── Шаг 0: найти n_e0 для панели 1 (H-ratio) ─────────────────────────
    print(f"\n=== Bisect n_e0 for H(0)/H(R) = {TARGET_RATIO_230:.3f} "
          f"(p=133 Pa) ===")
    ne0_H = bisect_ne_for_h_ratio(TARGET_RATIO_230)
    print(f"  -> n_e0_H = {ne0_H:.3e} m^-3\n")

    # ── Шаг 1: H(0)/H(R) vs давление ────────────────────────────────────
    print("=== Panel 1: H(0)/H(R) vs pressure ===")
    p_model, ratio_model = panel1_h_ratio_vs_pressure(ne0_H)
    print()

    # ── Шаг 2: j vs мощность (используем NE_J) ───────────────────────────
    print("=== Panel 2: j vs power ===")
    hw_model, j_model2, P_model2 = panel2_j_vs_power(NE_J)
    print()

    # ── Шаг 3: j vs расход (1D — горизонтальная линия) ───────────────────
    print("=== Panel 3: j vs flow ===")
    j_flat = panel3_j_vs_flow(NE_J)
    print()

    # ── Нормировка панели 2 ───────────────────────────────────────────────
    # Нормируем модель и эксперимент на значение при опорной точке:
    # модель: P_model ≈ P_ref → ближайший к H_wall=150 кА/м
    idx_ref = np.argmin(np.abs(hw_model - H_WALL_REF))
    j_ref_model = j_model2[idx_ref]
    P_ref_model = P_model2[idx_ref]

    # Эксперимент: опорная точка при P_p ≈ 2 кВт
    exp_235_P_ref = 2.0   # кВт
    exp_235_j_ref = float(np.interp(exp_235_P_ref, exp_235_P_kW, exp_235_j))

    j_model2_norm  = j_model2  / j_ref_model
    P_model2_norm  = P_model2  / P_ref_model
    j_exp235_norm  = exp_235_j / exp_235_j_ref
    P_exp235_norm  = exp_235_P_kW / exp_235_P_ref

    # Нормировка панели 3
    j_flat_norm   = j_flat   / j_ref_model
    exp_236_j_ref = float(np.mean(exp_236_j))
    j_exp236_norm  = exp_236_j / exp_236_j_ref

    # ── Layout 1×3 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(
        "Влияние давления, мощности и расхода газа на параметры ВЧИ-разряда\n"
        r"Аргон, $f = 1{,}76$ МГц, $R = 12$ мм",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13,
                        wspace=0.38)

    # ── [0] H(0)/H(R) vs давление ────────────────────────────────────────
    ax1 = axes[0]

    ax1.plot(exp_230_p, exp_230_ratio,
             "ks", ms=8, zorder=5,
             label=r"Эксп. Рис. 2.30, $z\!=\!-60$ мм (Ar)")

    # Маскируем NaN (неконвергентные точки при p < ~90 Па)
    mask_ok = ~np.isnan(ratio_model)
    ax1.plot(p_model[mask_ok], ratio_model[mask_ok],
             "#1f77b4", lw=2.0, marker="o", ms=5,
             label=rf"Модель 1D ($n_{{e0}}\!=\!{ne0_H:.1e}$ м$^{{-3}}$)")
    if np.any(~mask_ok):
        ax1.plot(p_model[~mask_ok], np.ones(np.sum(~mask_ok)) * 0.58,
                 "x", color="#1f77b4", ms=9, mew=2,
                 label="Неустойч. (δ ≪ h)")

    # Горизонтальная линия — среднее экспериментальное значение
    ax1.axhline(np.mean(exp_230_ratio), color="gray", lw=0.9, ls="--",
                alpha=0.6, label=f"Среднее эксп. = {np.mean(exp_230_ratio):.3f}")

    ax1.set_xlabel(r"$p$, Па")
    ax1.set_ylabel(r"$H_z(0)\,/\,H_z(R)$")
    ax1.set_title("Влияние давления\n"
                  r"$H_z(0)/H_z(R)$ vs $p$")
    ax1.set_xlim(0, 480)
    ax1.set_ylim(0.55, 1.05)
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.text(0.04, 0.97,
             r"Фикс. $n_{e0}$: рост $p$ → $\sigma_a \!\propto\! 1/\nu_c\!\downarrow$" + "\n"
             r"$\Rightarrow\delta\uparrow\Rightarrow$ скин ослабевает" + "\n"
             "Эксп. — слабая зависимость\n"
             r"(в реальности $n_e$ тоже меняется)",
             transform=ax1.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))

    # ── [1] j vs мощность (нормировано) ───────────────────────────────────
    ax2 = axes[1]

    ax2.plot(P_exp235_norm, j_exp235_norm,
             "ks", ms=8, zorder=5,
             label=r"Эксп. Рис. 2.35 (Ar, $z\!=\!-120$ мм, $G\!=\!0$)")

    ax2.plot(P_model2_norm, j_model2_norm,
             "#d62728", lw=2.0, marker="o", ms=5,
             label=rf"Модель 1D ($n_{{e0}}\!=\!{NE_J:.1e}$ м$^{{-3}}$)")

    # Степенная аппроксимация экспериментальных данных
    log_P = np.log(P_exp235_norm)
    log_j = np.log(j_exp235_norm)
    beta_exp = float(np.polyfit(log_P, log_j, 1)[0])
    P_fit = np.linspace(P_exp235_norm.min(), P_exp235_norm.max(), 50)
    j_fit = P_fit ** beta_exp
    ax2.plot(P_fit, j_fit, "--", color="gray", lw=1.0,
             label=rf"Эксп. $\propto P^{{{beta_exp:.2f}}}$")

    # Модельная степенная аппроксимация
    log_Pm = np.log(P_model2_norm)
    log_jm = np.log(j_model2_norm)
    beta_mod = float(np.polyfit(log_Pm, log_jm, 1)[0])
    P_mfit = np.linspace(P_model2_norm.min(), P_model2_norm.max(), 50)
    j_mfit = P_mfit ** beta_mod
    ax2.plot(P_mfit, j_mfit, ":", color="#d62728", lw=1.0,
             label=rf"Модель $\propto P^{{{beta_mod:.2f}}}$")

    ax2.set_xlabel(r"$P / P_{\mathrm{ref}}$  ($P_{\mathrm{ref}} = 2$ кВт)")
    ax2.set_ylabel(r"$j_\varphi\,/\,j_{\mathrm{ref}}$")
    ax2.set_title("Влияние мощности\n"
                  r"$j_\varphi$ vs $P_p$ (нормировано)")
    ax2.legend(loc="upper left", fontsize=8.5)

    # ── [2] j vs расход ───────────────────────────────────────────────────
    ax3 = axes[2]

    # Эксперимент (нормировано на среднее)
    ax3.plot(exp_236_G_sccm, j_exp236_norm,
             "ks", ms=8, zorder=5,
             label=r"Эксп. Рис. 2.36 (Ar, $z\!=\!-120$ мм)")

    # Горизонтальная линия модели — Pe≪1
    G_sccm_range = np.array([100, 7000])
    ax3.axhline(j_flat_norm, color="#1f77b4", lw=2.0,
                label=rf"Модель 1D: $j = \mathrm{{const}}$ (Pe $\ll$ 1)")

    # Заполнение ±10% как оценка неопределённости
    ax3.fill_between(G_sccm_range,
                     [j_flat_norm * 0.90] * 2,
                     [j_flat_norm * 1.10] * 2,
                     alpha=0.15, color="#1f77b4", label="±10% коридор")

    # Текущие расходы
    for Gs, lab in zip([500, 2000], ["500 sccm", "2000 sccm"]):
        ax3.axvline(Gs, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax3.text(Gs + 50, 0.62, lab, fontsize=7.5, color="gray", va="bottom")

    ax3.set_xscale("log")
    ax3.set_xlabel(r"$G$, sccm")
    ax3.set_ylabel(r"$j_\varphi\,/\,\langle j_\varphi^{\mathrm{exp}}\rangle$")
    ax3.set_title("Влияние расхода газа\n"
                  r"$j_\varphi$ vs $G$ (нормировано)")
    ax3.set_xlim(200, 8000)
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.text(0.04, 0.97,
             r"$\mathrm{Pe} = v_0 R / D_a \sim 5{\cdot}10^{-4}$–$7{\cdot}10^{-3}$" + "\n"
             "диффузия >> конвекция\n"
             "→ модель: $j$ не зависит от $G$",
             transform=ax3.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))

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
        path = os.path.join(out_dir, "compare_trends.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
