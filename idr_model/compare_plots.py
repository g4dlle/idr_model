"""
compare_plots.py — сравнение модели ВЧИ с экспериментальными данными.

Сравниваемые графики:
  • Рис. 2.27 кривая 3 — H_z(r)/H_z(R) по радиусу       (ВЧИ, Ar, G=0, z=−120 мм)
  • Рис. 2.33 кривая 3 — j_φ(r)/j_max  по радиусу       (ВЧИ, Ar, G=0, z=−120 мм)
  • 2D карты n_e(r,z) и σ_a(r,z) из 2D модели
  • Рис. 2.34 кривая 1 — j_φ(z)/j_max вдоль оси потока  (ВЧИ, Ar, G=0)

Позиции измерений:
  z_exp = −120 мм  ↔  z_model = L/2      (центр индуктора)
  z_exp = −60 мм   ↔  z_model = L/2 + 60 мм (полуширина распределения j)
  Ось z сдвинута: Δz = z_exp − z_exp_peak  / z_model − L_long/2

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

sys.path.insert(0, os.path.dirname(__file__))
from self_consistent import solve_maxwell_for_ne0
from solver_2d import solve_idr_2d
import config as cfg

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

R      = 0.012   # м, радиус трубки
L      = 0.05    # м, длина домена для радиальных профилей
L_LONG = 0.12    # м, длина домена для осевого профиля j(z)

# ════════════════════════════════════════════════════════════════════════════
# Экспериментальные данные (оцифровано из графиков)
# ════════════════════════════════════════════════════════════════════════════

# Рис. 2.27, кривая 3 — ВЧИ, Ar, G=0: H_z(r) [×10² А/м]
# Положение: z = −120 мм (центр индуктора)
exp_227_r  = np.array([0.19, 4.74, 8.64, 11.81]) * 1e-3   # м
exp_227_Hz = np.array([41.50, 41.64, 43.93, 54.59]) * 1e2  # А/м

# Рис. 2.33, кривая 3 — ВЧИ, Ar, G=0: j_φ(r) [×10⁶ А/м²]
# Положение: z = −120 мм (центр индуктора, подтверждено рис. 2.35 кривой 1)
exp_233_r = np.array([0.05, 2.46, 5.13, 8.02, 11.17]) * 1e-3  # м
exp_233_j = np.array([0.00, 0.39, 1.51, 1.82, 1.83]) * 1e6    # А/м²

# Рис. 2.34, кривая 1 — ВЧИ, Ar, G=0: j_φ(z) [×10⁶ А/м²]
# Пик при z_exp = −119.64 мм → сдвиг оси: z_shift = z_exp − z_exp_peak
_z234_exp = np.array([-176.87, -119.64, -61.10, 32.55])   # мм (исходные)
_j234_exp = np.array([  0.69,    1.88,   0.38,   0.00]) * 1e6  # А/м²
Z234_PEAK = _z234_exp[np.argmax(_j234_exp)]                # = −119.64 мм
exp_234_z = (_z234_exp - Z234_PEAK)                        # мм, сдвинуто к пику
exp_234_j = _j234_exp / np.max(_j234_exp)                  # нормировано

# ════════════════════════════════════════════════════════════════════════════
# Параметры модельных расчётов
# ════════════════════════════════════════════════════════════════════════════

P_PA = 133.0   # Па

# n_e0 для рис. 2.33 (δ >> R, j ∝ r):
NE_J = 3.71e17  # м⁻³

# Целевое отношение H(0)/H(R) из рис. 2.27
TARGET_H_RATIO = exp_227_Hz[0] / exp_227_Hz[-1]   # ≈ 0.760


# ════════════════════════════════════════════════════════════════════════════
# Вспомогательные расчёты
# ════════════════════════════════════════════════════════════════════════════

def _1d(ne0, N=100):
    """1D: возвращает r, H(r), j_φ(r)."""
    res = solve_maxwell_for_ne0(
        n_e0=ne0, N=N, R=R, p_pa=P_PA,
        H_wall=cfg.H_WALL, max_iter=500, tol=1e-5,
    )
    r = res["r"]
    H = np.sqrt(np.maximum(res["u"], 0.0))
    j = res["sigma_a"] * np.sqrt(np.maximum(res["v"], 0.0))
    return r, H, j


def find_ne_for_h_ratio(target_ratio, N=80,
                        ne_lo=1e19, ne_hi=1e23, n_iter=30):
    """Бисекцией находит n_e0 → H(0)/H(R) = target_ratio (1D)."""
    for _ in range(n_iter):
        ne_mid = np.sqrt(ne_lo * ne_hi)
        r, H, _ = _1d(ne_mid, N=N)
        ratio = H[0] / H[-1] if H[-1] > 0 else 1.0
        if ratio > target_ratio:
            ne_lo = ne_mid
        else:
            ne_hi = ne_mid
    return np.sqrt(ne_lo * ne_hi)


def find_ne_for_h_ratio_2d(target_ratio, L_dom=L, Nr=60, Nz=40,
                            ne_lo=1e19, ne_hi=1e23, n_iter=20):
    """Бисекцией находит n_e0 → H_mid(0)/H_mid(R) = target_ratio (2D, z=L/2)."""
    for i in range(n_iter):
        ne_mid = np.sqrt(ne_lo * ne_hi)
        r, _, H_mid, _, _ = _2d(ne_mid, L_dom=L_dom, Nr=Nr, Nz=Nz)
        ratio = H_mid[0] / H_mid[-1] if H_mid[-1] > 0 else 1.0
        print(f"  2D bisect [{i+1}/{n_iter}]: n_e={ne_mid:.3e}, H(0)/H(R)={ratio:.4f}")
        if ratio > target_ratio:
            ne_lo = ne_mid
        else:
            ne_hi = ne_mid
    return np.sqrt(ne_lo * ne_hi)


def _2d(ne0, L_dom=L, Nr=60, Nz=40):
    """2D: возвращает r, z, H_mid, j_mid и полный результат."""
    res = solve_idr_2d(
        Nr=Nr, Nz=Nz, R=R, L=L_dom,
        p_pa=P_PA, H_wall=cfg.H_WALL, n_e0=ne0,
        max_iter=300, tol=1e-3, relax=0.5,
        verbose=False,
    )
    r, z = res["r"], res["z"]
    jm    = len(z) // 2
    H_mid = np.sqrt(np.maximum(res["u"][:, jm], 0.0))
    j_mid = res["sigma_a"][:, jm] * np.sqrt(np.maximum(res["v"][:, jm], 0.0))
    return r, z, H_mid, j_mid, res


# ════════════════════════════════════════════════════════════════════════════
# Построение фигуры
# ════════════════════════════════════════════════════════════════════════════

def make_figure():
    # ── Расчёты ──────────────────────────────────────────────────────────
    print(f"Bisect n_e for H(0)/H(R) = {TARGET_H_RATIO:.3f}...")
    NE_H = find_ne_for_h_ratio(TARGET_H_RATIO)
    print(f"  n_e = {NE_H:.3e} m^-3")

    print("1D H-profile...")
    r1H, H1d, _ = _1d(NE_H)

    print("1D j-profile...")
    r1J, _, j1d = _1d(NE_J)

    print("2D (L=50mm, j-profile + maps)...")
    r2, z2, _, j2d, res2d = _2d(NE_J, L_dom=L)

    print(f"Bisect n_e for H(0)/H(R) = {TARGET_H_RATIO:.3f} (2D)...")
    NE_H_2D = find_ne_for_h_ratio_2d(TARGET_H_RATIO)
    print(f"  n_e (2D) = {NE_H_2D:.3e} m^-3")

    print("2D (L=50mm, H-profile)...")
    r2H, _, H2dH, _, _ = _2d(NE_H_2D, L_dom=L)

    print("2D (L=120mm, j(z) axial)...")
    r2z, z2z, _, _, res2z = _2d(NE_J, L_dom=L_LONG, Nr=60, Nz=60)

    print("Plotting...")

    # ── j_φ(z) из длинного 2D: берём r, где j максимален при z=L/2 ────────
    jm_long  = len(z2z) // 2
    j_at_mid = res2z["sigma_a"][:, jm_long] * np.sqrt(
        np.maximum(res2z["v"][:, jm_long], 0.0))
    ir_max   = int(np.argmax(j_at_mid))   # индекс r с максимальным j при z=L/2

    j_axial  = res2z["sigma_a"][ir_max, :] * np.sqrt(
        np.maximum(res2z["v"][ir_max, :], 0.0))
    j_axial_norm = j_axial / np.max(j_axial) if np.max(j_axial) > 0 else j_axial

    # Сдвиг z модели: пик стоит при z = L_LONG/2; смещаем к нулю
    z_model_shifted = (z2z - L_LONG / 2) * 1e3   # мм, 0 = центр индуктора

    # ── Layout 3×2 ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(
        "Сравнение модели ВЧИ с экспериментом\n"
        r"Аргон, $f = 1{,}76$ МГц, $R = 12$ мм, $p = 133$ Па",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.08, right=0.97,
        top=0.93, bottom=0.06,
        hspace=0.42, wspace=0.34,
        height_ratios=[1, 1, 1],
    )

    # ── [0,0] H_z(r)/H_z(R) — Рис. 2.27 кривая 3 ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    Hz_exp_norm = exp_227_Hz / exp_227_Hz[-1]
    H1_norm     = H1d / H1d[-1]
    H2_norm     = H2dH / H2dH[-1] if H2dH[-1] > 0 else H2dH

    ax1.plot(exp_227_r * 1e3, Hz_exp_norm,
             "ks", ms=8, zorder=5, label="Эксп. Рис. 2.27 (ВЧИ, Ar, G=0)")
    ax1.plot(r1H * 1e3, H1_norm, "#1f77b4", lw=2.0,
             label=rf"1D ($n_{{e0}}={NE_H:.2e}$ м$^{{-3}}$)")
    ax1.plot(r2H * 1e3, H2_norm, "#d62728", lw=2.0, ls="--",
             label=rf"2D ($n_{{e0}}={NE_H_2D:.2e}$ м$^{{-3}}$, $z=L/2$)")

    ax1.set_xlabel("$r$, мм")
    ax1.set_ylabel(r"$H_z(r)\,/\,H_z(R)$")
    ax1.set_title("Профиль $H_z(r)/H_z(R)$\n"
                  "(Рис. 2.27, кривая 3: ВЧИ, Ar, G=0, $z=-120$ мм)")
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0.60, 1.10)
    ax1.legend()

    # ── [0,1] j_φ(r)/j_max — Рис. 2.33 кривая 3 ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    j_exp_norm = exp_233_j / np.max(exp_233_j)
    j1_norm    = j1d / np.max(j1d) if np.max(j1d) > 0 else j1d
    j2_norm    = j2d / np.max(j2d) if np.max(j2d) > 0 else j2d

    ax2.plot(exp_233_r * 1e3, j_exp_norm,
             "ks", ms=8, zorder=5, label="Эксп. Рис. 2.33 (ВЧИ, Ar, G=0)")
    ax2.plot(r1J * 1e3, j1_norm, "#1f77b4", lw=2.0,
             label=rf"1D ($n_{{e0}}={NE_J:.1e}$ м$^{{-3}}$)")
    ax2.plot(r2 * 1e3, j2_norm, "#d62728", lw=2.0, ls="--",
             label="2D (z = L/2)")

    ax2.set_xlabel("$r$, мм")
    ax2.set_ylabel(r"$j_\varphi(r)\,/\,j_{\varphi,\max}$")
    ax2.set_title(r"Профиль $j_\varphi(r)/j_{\varphi,\max}$" + "\n"
                  "(Рис. 2.33, кривая 3: ВЧИ, Ar, G=0, $z=-120$ мм)")
    ax2.set_xlim(0, 12)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend()

    # ── [1,0] 2D карта n_e(r,z) ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    Z2, R2 = np.meshgrid(z2 * 1e3, r2 * 1e3)
    cf3 = ax3.contourf(Z2, R2, res2d["n_e"], levels=25, cmap="plasma")
    fig.colorbar(cf3, ax=ax3, label=r"$n_e$, м$^{-3}$")
    ax3.set_xlabel("$z$, мм")
    ax3.set_ylabel("$r$, мм")
    ax3.set_title(r"$n_e(r,z)$ — 2D модель, $L=50$ мм")

    # ── [1,1] 2D карта σ_a(r,z) ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    cf4 = ax4.contourf(Z2, R2, res2d["sigma_a"], levels=25, cmap="viridis")
    fig.colorbar(cf4, ax=ax4, label=r"$\sigma_a$, См/м")
    ax4.set_xlabel("$z$, мм")
    ax4.set_ylabel("$r$, мм")
    ax4.set_title(r"$\sigma_a(r,z)$ — 2D модель, $L=50$ мм")

    # ── [2,:] j_φ(z)/j_max — Рис. 2.34 кривая 1 ─────────────────────────
    ax5 = fig.add_subplot(gs[2, :])

    ax5.plot(exp_234_z, exp_234_j,
             "ks", ms=9, zorder=5, label="Эксп. Рис. 2.34 (ВЧИ, Ar, G=0)")

    ax5.plot(z_model_shifted, j_axial_norm,
             "#d62728", lw=2.2,
             label=rf"2D модель, $L={L_LONG*1e3:.0f}$ мм, $r={r2z[ir_max]*1e3:.1f}$ мм")

    # Вертикальные метки соответствия z позиций
    for z_label, name in [(-57.2, "$z=-177$ мм"), (0, "$z=-120$ мм (пик)"),
                           (58.5, "$z=-61$ мм")]:
        ax5.axvline(z_label, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax5.text(z_label + 1, 0.97, name, fontsize=8, color="gray",
                 va="top", rotation=90)

    ax5.set_xlabel("$z - z_{\\mathrm{пик}}$, мм  (0 = центр индуктора, $z_{\\mathrm{exp}}=-120$ мм)")
    ax5.set_ylabel(r"$j_\varphi(z)\,/\,j_{\varphi,\max}$")
    ax5.set_title(r"Осевой профиль $j_\varphi(z)/j_{\varphi,\max}$"
                  "\n(Рис. 2.34, кривая 1: ВЧИ, Ar, G=0)")
    ax5.set_xlim(-80, 160)
    ax5.set_ylim(-0.05, 1.15)
    ax5.legend(loc="upper right")

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
        print(f"Saved: {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
