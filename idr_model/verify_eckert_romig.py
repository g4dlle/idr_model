"""
verify_eckert_romig.py — верификационные таблицы для §3.5.

Тест 1 (Эккерт): Maxwell с однородной σ̃ = const.
  Аналитика: H(r) = H_wall · J₀(kr)/J₀(kR),  k² = iωμ₀σ̃ (комплексное).
  Численно:  прямое решение комплексной трёхдиагональной СЛАУ.

Тест 2 (2D, основа для ромиговского предела Pe→0):
  Однородные Da=D, νi=ν, Дирихле на всех 4 границах.
  Аналитика: λ₀² = νi / (Da·[(j₀₁/R)²+(π/L)²])
  Таблица сходимости O(h²) и предел L→∞ → 1D.

Запуск (из папки idr_model/):
    python verify_eckert_romig.py
    python verify_eckert_romig.py --save   # сохранить PNG в ../plots/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import jv, iv   # Bessel J_ν и I_ν (принимают комплексный z)

sys.path.insert(0, os.path.dirname(__file__))

from physics import conductivity, collision_freq
from equations_2d import make_grid_2d
from self_consistent_2d import compute_lambda0_2d
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

MU_0 = 4.0e-7 * np.pi  # Гн/м


# ===========================================================================
# Тест 1 — Эккерт: Maxwell с однородной σ̃
# ===========================================================================

def _I0c(z):
    """I₀(z) = J₀(iz) — модифицированная функция Бесселя 1-го рода, order 0."""
    return iv(0, z)


def solve_maxwell_complex(N, R, k_sq, H_wall, r_min=0.0):
    """
    Решение (1/r)d/dr(r·dH/dr) - k²·H = 0  с k² = iωμ₀σ̃ (комплексное).
    Возвращает массив H[0..N] (комплексный).
    BC: H(R) = H_wall,  dH/dr|_{r_min} = 0.
    """
    r = np.linspace(r_min, R, N + 1)
    h = (R - r_min) / N

    n = N + 1
    lower = np.zeros(n, dtype=complex)
    main  = np.zeros(n, dtype=complex)
    upper = np.zeros(n, dtype=complex)
    rhs   = np.zeros(n, dtype=complex)

    annular = r_min > 0.0

    # Граничное условие на правой стенке
    main[-1] = 1.0
    rhs[-1]  = H_wall

    # Граничное условие на левой границе
    if annular:
        # Включение: H(r_min) = H_wall (обе стенки несут одинаковое поле)
        main[0] = 1.0
        rhs[0]  = H_wall
    else:
        # Ось: симметрия (L'Hôpital): 2(H₁-H₀)/h² - k²H₀ = 0
        main[0]  = -2.0 / h**2 - k_sq
        upper[0] =  2.0 / h**2
        rhs[0]   = 0.0

    # Внутренние узлы i = 1 .. N-1
    for i in range(1, N):
        ri     = r[i]
        r_half_p = ri + h / 2.0   # r_{i+1/2}
        r_half_m = ri - h / 2.0   # r_{i-1/2}
        coef = 1.0 / (ri * h**2)
        lower[i] = coef * r_half_m
        main[i]  = -coef * (r_half_p + r_half_m) - k_sq
        upper[i] = coef * r_half_p
        rhs[i]   = 0.0

    # Решение методом прогонки (комплексная арифметика)
    c = upper.copy()
    d = rhs.copy()
    m = main.copy()

    c[0] /= m[0]
    d[0] /= m[0]
    for i in range(1, n):
        denom = m[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (d[i] - lower[i] * d[i - 1]) / denom

    H = np.zeros(n, dtype=complex)
    H[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        H[i] = d[i] - c[i] * H[i + 1]

    return r, H


def eckert_test(N=200, n_e0=6.2e20, p_pa=133.0, R=0.012, H_wall=150e3,
                n_points_table=10, plot=True):
    """
    Верификация Maxwell-блока по Эккерту (стандартная геометрия).
    Возвращает: r, H_num/H_wall, H_an/H_wall, rel_err, k (комплексное).
    """
    sigma_a, sigma_p, _ = conductivity(n_e0, p_pa)
    sigma_tilde = sigma_a - 1j * sigma_p          # комплексная проводимость
    k_sq  = 1j * cfg.OMEGA * MU_0 * sigma_tilde   # k² = iωμ₀σ̃
    k     = np.sqrt(k_sq)                          # выбираем ветвь с Im(k)>0

    r, H_num = solve_maxwell_complex(N, R, k_sq, H_wall)

    # Аналитическое решение: уравнение (1/r)d/dr(r dH/dr) = k²H → I₀(kr)/I₀(kR)
    # k² = iωμ₀σ̃ → k = sqrt(k²), решение - модифицированные бесселевые I₀, K₀
    H_an = H_wall * _I0c(k * r) / _I0c(k * R)

    H_num_norm = np.abs(H_num) / H_wall
    H_an_norm  = np.abs(H_an)  / H_wall
    rel_err    = np.abs(H_num_norm - H_an_norm) / np.maximum(H_an_norm, 1e-30)

    # Параметры скин-слоя
    delta = np.sqrt(2.0 / (cfg.OMEGA * MU_0 * sigma_a))

    print("\n" + "="*65)
    print("ТЕСТ 1 — ЭККЕРТ: Maxwell с однородной σ̃ = const")
    print("="*65)
    print(f"  p = {p_pa:.0f} Па,  R = {R*1e3:.0f} мм,  n_e0 = {n_e0:.2e} м⁻³")
    print(f"  σ_a = {sigma_a:.4e} См/м,  σ_p = {sigma_p:.4e} См/м")
    print(f"  |k| = {abs(k):.3f} м⁻¹,  δ = {delta*1e3:.2f} мм")
    print(f"  δ/R = {delta/R:.3f}")
    print(f"  N = {N} узлов (h = {R/N*1e6:.1f} мкм)")
    print()

    idx = np.round(np.linspace(0, N, n_points_table)).astype(int)
    header = f"  {'r/R':>6}  {'|H_num|/H_w':>13}  {'|H_an|/H_w':>12}  {'err, %':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in idx:
        print(f"  {r[i]/R:6.3f}  {H_num_norm[i]:13.6f}  {H_an_norm[i]:12.6f}  {rel_err[i]*100:8.4f}")

    max_err = np.max(rel_err[1:])  # исключаем стенку (там =0 по построению)
    print(f"\n  Максимальная относительная погрешность: {max_err*100:.4f} %")

    return r, H_num_norm, H_an_norm, rel_err, k, delta, sigma_a, sigma_p


def eckert_grid_convergence(n_e0=6.2e20, p_pa=133.0, R=0.012, H_wall=150e3):
    """Таблица сходимости O(h²) для Maxwell-блока."""
    sigma_a, sigma_p, _ = conductivity(n_e0, p_pa)
    k_sq = 1j * cfg.OMEGA * MU_0 * (sigma_a - 1j * sigma_p)
    k    = np.sqrt(k_sq)

    grids = [25, 50, 100, 200, 400]
    errors = []
    for N in grids:
        r, H_num = solve_maxwell_complex(N, R, k_sq, H_wall)
        H_an = H_wall * _I0c(k * r) / _I0c(k * R)
        err = np.max(np.abs(np.abs(H_num) - np.abs(H_an)) / H_wall)
        errors.append(err)

    print("\n  Сходимость Maxwell-решателя (Эккерт)")
    print(f"  {'N':>5}  {'h, мкм':>8}  {'max|err|':>12}  {'ratio':>7}")
    print("  " + "-" * 40)
    for i, (N, err) in enumerate(zip(grids, errors)):
        h_mm = R / N * 1e6
        ratio = errors[i - 1] / err if i > 0 else float("nan")
        print(f"  {N:5d}  {h_mm:8.1f}  {err:12.4e}  {ratio:7.2f}")

    return grids, errors


# ===========================================================================
# Тест 2 — 2D: сходимость и предел L→∞ (основа Pe→0 по Ромигу)
# ===========================================================================

J01 = 2.4048255577


def romig_grid_convergence_test():
    """
    Таблица 1: сходимость λ₀² при удвоении сетки.
    Точное решение: λ₀² = νi/(Da·[(j₀₁/R)²+(π/L)²])  при Da=νi=R=L=1.
    """
    R, L = 1.0, 1.0
    D, nu = 1.0, 1.0
    lam_exact = nu / (D * ((J01 / R)**2 + (np.pi / L)**2))

    grids = [(20, 10), (40, 20), (80, 40), (160, 80)]
    rows  = []
    for Nr, Nz in grids:
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
        Da   = np.full((Nr + 1, Nz + 1), D)
        nu_i = np.full((Nr + 1, Nz + 1), nu)
        lam = compute_lambda0_2d(r, z, hr, hz, Da, nu_i,
                                 bc_z_sigma="dirichlet")
        rows.append((Nr, Nz, lam, abs(lam - lam_exact), abs(lam - lam_exact) / lam_exact))

    print("\n" + "="*65)
    print("ТЕСТ 2 — 2D: сходимость λ₀² O(h²)  [Da=νi=R=L=1, Дирихле]")
    print(f"  λ₀²_точн = {lam_exact:.6f}")
    print("="*65)
    print(f"  {'Nr':>5}  {'Nz':>4}  {'λ₀²_чис':>10}  {'|ошибка|':>10}  {'отн., %':>8}  {'ratio':>6}")
    print("  " + "-" * 52)
    for i, (Nr, Nz, lam, err, rel) in enumerate(rows):
        ratio = rows[i - 1][3] / err if i > 0 else float("nan")
        print(f"  {Nr:5d}  {Nz:4d}  {lam:10.6f}  {err:10.2e}  {rel*100:8.4f}  {ratio:6.2f}")

    return rows, lam_exact


def romig_large_L_test():
    """
    Таблица 2: предел L→∞ (Neumann) — λ₀² стремится к 1D-значению.
    При Дирихле: λ₀² = νi/(Da·[(j₀₁/R)²+(π/L)²]) → 1D = R²·νi/(Da·j₀₁²) снизу.
    """
    R   = 1.0
    D, nu = 1.0, 1.0
    j01 = J01
    lam_1d = nu * R**2 / (D * j01**2)

    L_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    Nr = 80
    rows = []
    for L in L_values:
        lam_exact = nu / (D * ((j01 / R)**2 + (np.pi / L)**2))
        Nz = max(10, int(Nr * L / R / 2))
        r, z, hr, hz = make_grid_2d(Nr, Nz, R, L)
        Da_arr   = np.full((Nr + 1, Nz + 1), D)
        nu_arr   = np.full((Nr + 1, Nz + 1), nu)
        lam = compute_lambda0_2d(r, z, hr, hz, Da_arr, nu_arr,
                                 bc_z_sigma="dirichlet")
        rel_to_1d = (lam_1d - lam) / lam_1d
        rows.append((L, Nz, lam, lam_exact, rel_to_1d))

    print("\n" + "="*65)
    print(f"ТЕСТ 2b — предел L/R→∞:  λ₀² → 1D = {lam_1d:.6f}")
    print("="*65)
    print(f"  {'L/R':>5}  {'Nz':>4}  {'λ₀²_чис':>10}  {'λ₀²_точн':>10}  {'отклон.от 1D, %':>16}")
    print("  " + "-" * 56)
    for L, Nz, lam, lam_exact, rel in rows:
        print(f"  {L:5.0f}  {Nz:4d}  {lam:10.6f}  {lam_exact:10.6f}  {rel*100:16.3f}")

    return rows, lam_1d


# ===========================================================================
# Построение графиков
# ===========================================================================

def make_figure(eckert_data, grid_data, L_data, lam_exact, lam_1d, save_dir=None):
    r_arr, H_num, H_an, rel_err, k, delta, sigma_a, sigma_p = eckert_data
    R = r_arr[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Верификация: тест Эккерта (1D Maxwell) и 2D сходимость",
                 fontsize=13, fontweight="bold")

    # ── [0] Эккерт: профили H ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(r_arr / R, H_an,  "#1f77b4", lw=2.5, label="Аналитика: $|I_0(kr)|/|I_0(kR)|$")
    ax.plot(r_arr / R, H_num, "#d62728", lw=1.5, ls="--",
            label=f"Численно: N = {len(r_arr)-1}")
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("$|H_z(r)|\\ /\\ H_\\mathrm{wall}$")
    ax.set_title(f"Эккерт: однородная $\\tilde{{\\sigma}}$\n"
                 f"$\\delta/R = {delta/R:.2f}$,  "
                 f"$\\sigma_a = {sigma_a:.2e}$ См/м")
    ax.legend()

    # ── [1] Эккерт: относительная погрешность вдоль r ────────────────────
    ax = axes[1]
    ax.semilogy(r_arr / R, np.maximum(rel_err, 1e-16) * 100, "#2ca02c", lw=2.0)
    ax.set_xlabel("$r/R$")
    ax.set_ylabel("Относительная погрешность, %")
    ax.set_title("Погрешность Maxwell-решателя (Эккерт)")

    # ── [2] 2D: сходимость λ₀² и предел L→∞ ──────────────────────────────
    ax = axes[2]
    Nr_vals = [row[0] for row in grid_data]
    lam_vals = [row[2] for row in grid_data]
    ax.semilogx(Nr_vals, lam_vals, "s-", color="#ff7f0e", lw=2.0, ms=8,
                label="$\\lambda_0^2$, числ. (2D)")
    ax.axhline(lam_exact, color="#1f77b4", lw=1.5, ls="--",
               label=f"Точн. (Дирихле): {lam_exact:.4f}")
    ax.axhline(lam_1d, color="#9467bd", lw=1.5, ls=":",
               label=f"1D-предел: {lam_1d:.4f}")
    ax.set_xlabel("Nr (число узлов по $r$)")
    ax.set_ylabel("$\\lambda_0^2$")
    ax.set_title("2D: сходимость $\\lambda_0^2$ при $D_a=\\nu_i=1$")
    ax.legend()

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "fig_verification_eckert_2d.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nСохранено: {path}")
    else:
        plt.show()

    return fig


# ===========================================================================
# main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", nargs="?", const="../plots", metavar="DIR")
    args = parser.parse_args()

    # Тест 1: Эккерт
    eckert_data = eckert_test(N=200, n_e0=6.2e20)
    grids, errors = eckert_grid_convergence(n_e0=6.2e20)

    # Тест 2: 2D сходимость и предел L→∞
    grid_rows, lam_exact = romig_grid_convergence_test()
    L_rows, lam_1d       = romig_large_L_test()

    # Рисунок
    make_figure(eckert_data, grid_rows, L_rows, lam_exact, lam_1d,
                save_dir=args.save)


if __name__ == "__main__":
    main()
