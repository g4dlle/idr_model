"""
run_diploma_results.py — скрипт генерации результатов для дипломной работы.

Производит:
  Таблица 1 — n_e0* при разных давлениях и радиусах трубки (параметрический анализ)
  Таблица 2 — влияние проводящего включения на n_e0* и мощность
  Таблица 3 — параметры течения Пуазейля при разных расходах газа
  Рис. П1   — профиль скорости Пуазейля v(r) для трёх расходов G
  Рис. П2   — совмещённый профиль v(r)/v0 и n_e(r)/n_e0 (1D)

Таблицы сохраняются в plots/ в формате CSV и выводятся в терминал.
Графики сохраняются как PNG.

Использование
-------------
  cd idr_model
  python run_diploma_results.py              # показать графики интерактивно
  python run_diploma_results.py --save       # сохранить в plots/
"""

import sys
import os
import argparse
import csv

# Принудительно UTF-8 для stdout (Windows cp1251 не поддерживает греческие символы)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from self_consistent import find_n_e0, solve_self_consistent, solve_maxwell_for_ne0
from solver import solve_idr
from physics import (
    poiseuille_velocity, poiseuille_v0,
    sccm_to_kg_s, gas_mass_density,
)
from config import P_PA, R_TUBE, H_WALL, N_GRID, T_NEUTRAL

# ── Стиль графиков (единый для диплома) ─────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "lines.linewidth":  1.8,
})

# ── Вспомогательные функции ───────────────────────────────────────────────────

def _savefig(fig, fpath):
    """Сохранить или показать фигуру."""
    if fpath:
        fig.savefig(fpath, bbox_inches="tight")
        print(f"  saved: {fpath}")
        plt.close(fig)
    else:
        plt.show()


def _print_table(headers, rows, title=""):
    """Красивый вывод таблицы в терминал."""
    if title:
        print(f"\n{'-'*60}")
        print(f"  {title}")
        print(f"{'-'*60}")
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt.format(*row))
    print()


def _save_csv(fpath, headers, rows):
    """Сохранить таблицу в CSV."""
    if fpath is None:
        return
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  CSV: {fpath}")


# ── Вспомогательная функция: сканирование λ₀(n_e0) ──────────────────────────

def scan_lambda_ne0(p_pa, R, H_wall=H_WALL, n_pts=18):
    """
    Вычисляет λ₀ на log-равномерной сетке по n_e0 ∈ [1e14, 1e22].

    Returns
    -------
    ne0_arr  : (n_pts,) log-равномерная сетка n_e0
    lam0_arr : (n_pts,) соответствующие λ₀
    """
    ne0_arr  = np.logspace(14, 22, n_pts)
    lam0_arr = np.zeros(n_pts)
    for i, ne0 in enumerate(ne0_arr):
        sol = solve_maxwell_for_ne0(
            n_e0=ne0, N=N_GRID, R=R, p_pa=p_pa, H_wall=H_wall,
            max_iter=300, tol=1e-5, relax=0.5,
        )
        lam0_arr[i] = sol["lambda0"]
    return ne0_arr, lam0_arr


def _find_crossing(ne0_arr, lam0_arr):
    """
    Находит n_e0* — точку пересечения λ₀(n_e0) = 1 методом логарифмической
    интерполяции между соседними точками, меняющими знак (λ₀ - 1).

    Returns (n_e0_star, lam0_star) или (None, None) если пересечения нет.
    """
    f = lam0_arr - 1.0
    for i in range(len(f) - 1):
        if f[i] * f[i + 1] <= 0.0:
            # Логарифмическая интерполяция по n_e0
            log_lo = np.log10(ne0_arr[i])
            log_hi = np.log10(ne0_arr[i + 1])
            # Линейная интерп по f
            t = f[i] / (f[i] - f[i + 1])
            log_star = log_lo + t * (log_hi - log_lo)
            ne0_star = 10.0**log_star
            lam0_star = 1.0 + f[i] * (1.0 - t) + (f[i + 1]) * t  # ≈ 1
            return ne0_star, lam0_star
    return None, None


# ── Таблица 1: параметрический анализ (p, R) → n_e0* ─────────────────────────

def table_parametric(save_dir):
    """
    Сканирование λ₀(n_e0) на log-сетке для каждой пары (p, R),
    затем интерполяция пересечения λ₀ = 1.
    Надёжнее бисекции: работает даже когда bracket условие не выполняется.
    """
    p_values = [66.5, 133.0, 266.0]     # Па (0.5, 1, 2 Торр)
    R_values = [0.008, 0.012, 0.018]    # м (8, 12, 18 мм)

    headers = ["p, Pa", "p, Torr", "R, mm",
               "n_e0*, m^-3", "E_R, V/m", "lam0_at_cross", "found"]
    rows = []

    for p_pa in p_values:
        for R in R_values:
            print(f"  p={p_pa:.1f} Pa, R={R*1e3:.0f} mm  scanning ...",
                  end=" ", flush=True)
            ne0_arr, lam0_arr = scan_lambda_ne0(p_pa, R, H_WALL, n_pts=18)
            ne0_star, lam0_star = _find_crossing(ne0_arr, lam0_arr)

            if ne0_star is not None:
                # Уточняем E_R при найденном n_e0*
                sol = solve_maxwell_for_ne0(
                    n_e0=ne0_star, N=N_GRID, R=R, p_pa=p_pa, H_wall=H_WALL,
                    max_iter=300, tol=1e-5, relax=0.5,
                )
                E_R = float(np.sqrt(max(sol["v"][-1], 0.0)))
                found = "yes"
                print(f"n_e0* = {ne0_star:.2e}  lam0 = {lam0_star:.4f}")
            else:
                # Нет пересечения — сообщаем крайнее значение
                idx_best = int(np.argmin(np.abs(lam0_arr - 1.0)))
                ne0_star  = ne0_arr[idx_best]
                lam0_star = lam0_arr[idx_best]
                sol = solve_maxwell_for_ne0(
                    n_e0=ne0_star, N=N_GRID, R=R, p_pa=p_pa, H_wall=H_WALL,
                    max_iter=300, tol=1e-5, relax=0.5,
                )
                E_R   = float(np.sqrt(max(sol["v"][-1], 0.0)))
                found = "no (closest)"
                print(f"no crossing, closest: n_e0={ne0_star:.2e}  lam0={lam0_star:.4f}")

            rows.append([
                f"{p_pa:.1f}",
                f"{p_pa/133.322:.3f}",
                f"{R*1e3:.0f}",
                f"{ne0_star:.3e}",
                f"{E_R:.1f}",
                f"{lam0_star:.4f}",
                found,
            ])

    _print_table(headers, rows, "Table 1 - n_e0* at different p and R")
    csv_path = os.path.join(save_dir, "table1_parametric.csv") if save_dir else None
    _save_csv(csv_path, headers, rows)
    return rows


# ── Таблица 2: влияние проводящего включения ──────────────────────────────────

def table_inclusion(save_dir, use_2d=False):
    """
    Сравнение разряда с включением и без.

    use_2d=False : solve_idr 1D при фиксированном n_e0 (быстро)
    use_2d=True  : solve_idr_2d при фиксированном n_e0 (медленнее, полная 2D)
    """
    from config import N_E0, N_Z, L_TUBE, BC_Z_SIGMA
    r_inc_values = [0.0, 0.002, 0.004, 0.006]   # м
    model_tag = "2D" if use_2d else "1D"

    if use_2d:
        from solver_2d import solve_idr_2d

    headers = ["r_inc, mm", f"sigma_max ({model_tag}), S/m", "n_e_max, m^-3",
               "Q_total, W/m", "Dsigma, %", "DQ, %"]
    rows = []

    baseline_sigma = None
    baseline_Q     = None

    for r_inc in r_inc_values:
        print(f"  [{model_tag}] r_inc={r_inc*1e3:.0f} mm  ...", end=" ", flush=True)

        if use_2d:
            sol = solve_idr_2d(
                Nr=N_GRID, Nz=N_Z, R=R_TUBE, L=L_TUBE,
                p_pa=P_PA, H_wall=H_WALL, n_e0=N_E0,
                r_inc=r_inc, bc_z_sigma=BC_Z_SIGMA,
                max_iter=500, tol=1e-6, relax=0.5,
            )
            sigma_max = float(np.max(sol["sigma_a"]))
            ne_max    = float(np.max(sol["n_e"]))
            r_arr     = sol["r"]
            z_arr     = sol["z"]
            r_col     = r_arr[:, np.newaxis]
            # Q = pi * integral(sigma_a * |E|^2 * r dr dz) / L  — на ед. длины
            Q_total = float(np.pi / (z_arr[-1] - z_arr[0]) * np.trapezoid(
                np.trapezoid(sol["sigma_a"] * sol["v"] * r_col, z_arr, axis=1),
                r_arr))
        else:
            sol = solve_idr(
                N=N_GRID, R=R_TUBE, p_pa=P_PA, H_wall=H_WALL,
                n_e0=N_E0, r_inc=r_inc,
                max_iter=500, tol=1e-6, relax=0.5,
            )
            sigma_max = float(np.max(sol["sigma_a"]))
            ne_max    = float(np.max(sol["n_e"]))
            r_arr     = sol["r"]
            Q_total   = float(np.pi * np.trapezoid(
                sol["sigma_a"] * sol["v"] * r_arr, r_arr))

        if r_inc == 0.0:
            baseline_sigma = sigma_max
            baseline_Q     = Q_total
            dsig = "baseline"
            dQ   = "baseline"
        else:
            dsig = f"{(sigma_max - baseline_sigma) / baseline_sigma * 100:+.1f}"
            dQ   = f"{(Q_total   - baseline_Q)    / baseline_Q    * 100:+.1f}"

        rows.append([
            f"{r_inc*1e3:.0f}",
            f"{sigma_max:.4f}",
            f"{ne_max:.3e}",
            f"{Q_total:.4f}",
            dsig,
            dQ,
        ])
        print(f"sigma_max={sigma_max:.4f}  Q={Q_total:.4f}  conv={sol['converged']}")

    title = f"Table 2 - effect of conductive inclusion ({model_tag})"
    _print_table(headers, rows, title)
    csv_path = os.path.join(save_dir, "table2_inclusion.csv") if save_dir else None
    _save_csv(csv_path, headers, rows)
    return rows


# ── Таблица 3: параметры течения Пуазейля ─────────────────────────────────────

def table_poiseuille(save_dir):
    """Скорость Пуазейля для разных расходов при стандартных условиях."""
    G_values_sccm = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    p_pa = P_PA
    T_a  = T_NEUTRAL
    R    = R_TUBE

    rho = gas_mass_density(p_pa, T_a)
    r_half = R / 2.0

    headers = ["G, sccm", "G, kg/s", "v0, m/s", "v(R/2), m/s", "Re"]
    rows = []

    for G_sccm in G_values_sccm:
        G_kg_s = sccm_to_kg_s(G_sccm)
        v0     = poiseuille_v0(G_kg_s, R, p_pa, T_a)
        v_half = v0 * (1.0 - (r_half / R)**2)
        # Число Рейнольдса для трубы: Re = ρ·v_ср·D/μ,
        # v_ср = v₀/2 (для Пуазейля), D=2R, μ_Ar ≈ 2.27e-5 Па·с при 300 К
        mu_ar  = 2.27e-5
        v_mean = v0 / 2.0
        Re     = rho * v_mean * (2 * R) / mu_ar

        rows.append([
            f"{G_sccm:.0f}",
            f"{G_kg_s:.3e}",
            f"{v0:.4f}",
            f"{v_half:.4f}",
            f"{Re:.1f}",
        ])

    _print_table(headers, rows,
                 f"Table 3 - Poiseuille (p={p_pa:.0f} Pa, T={T_a:.0f} K, R={R*1e3:.0f} mm)")
    csv_path = os.path.join(save_dir, "table3_poiseuille.csv") if save_dir else None
    _save_csv(csv_path, headers, rows)
    return rows


# ── Рис. П1: профили v(r) при разных G ────────────────────────────────────────

def plot_poiseuille_profiles(save_path):
    """Профиль скорости Пуазейля v(r) для трёх значений G."""
    G_sccm_list = [100.0, 500.0, 2000.0]
    colors      = ["#2196F3", "#FF5722", "#4CAF50"]

    Nr = 200
    r  = np.linspace(0.0, R_TUBE, Nr + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Профиль скорости газа. Течение Пуазейля", fontsize=13)

    ax_abs, ax_norm = axes

    for G_sccm, color in zip(G_sccm_list, colors):
        G_kg_s = sccm_to_kg_s(G_sccm)
        v      = poiseuille_velocity(r, G_kg_s, R_TUBE, P_PA, T_NEUTRAL)
        v0     = v[0]
        label  = f"G = {G_sccm:.0f} sccm"

        ax_abs.plot(r * 1e3, v, color=color, label=label)
        ax_norm.plot(r * 1e3, v / v0, color=color, label=label)

    ax_abs.set_xlabel("r, мм")
    ax_abs.set_ylabel("v(r), м/с")
    ax_abs.set_title("Абсолютные скорости")
    ax_abs.legend()
    ax_abs.grid(True, alpha=0.3)
    ax_abs.set_xlim(0, R_TUBE * 1e3)
    ax_abs.set_ylim(bottom=0)

    ax_norm.set_xlabel("r, мм")
    ax_norm.set_ylabel(r"$v(r)\,/\,v_0$")
    ax_norm.set_title("Нормированный профиль")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)
    ax_norm.set_xlim(0, R_TUBE * 1e3)
    ax_norm.set_ylim(0, 1.05)

    # Аннотация: правильная формула расхода
    ax_norm.text(
        0.97, 0.55,
        r"$G = \frac{\pi}{2}\,\rho\,v_0\,R^2$"
        "\n"
        r"$v_0 = \frac{2G}{\pi\,\rho\,R^2}$",
        transform=ax_norm.transAxes,
        ha="right", va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    _savefig(fig, save_path)


# ── Рис. П2: v(r)/v0 совместно с n_e(r)/n_e0 ─────────────────────────────────

def plot_velocity_vs_density(save_path):
    """
    Совмещает нормированный профиль скорости и профиль концентрации электронов
    (из 1D самосогласованного решения).
    Показывает: плазма сосредоточена у стенки (скин-эффект),
    газ течёт быстрее по оси — физически обратные профили.
    """
    print("  Запуск самосогласованного решения для Рис. П2 ...", flush=True)
    sc = solve_self_consistent(
        N=N_GRID, R=R_TUBE, p_pa=P_PA, H_wall=H_WALL,
        tol_lambda=0.01, max_bisect=50, verbose=False,
    )

    sol   = sc["solution"]
    r     = sol["r"]
    n_e   = sol["n_e"]
    n_e0  = float(np.max(n_e))
    G_sccm_list = [100.0, 500.0, 2000.0]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Профиль концентрации электронов
    ax.plot(r * 1e3, n_e / n_e0,
            color="black", lw=2.2, ls="-",
            label=r"$n_e(r)\,/\,n_{e0}$  (плазма)")

    # Нормированные профили Пуазейля
    for G_sccm, color in zip(G_sccm_list, colors):
        G_kg_s = sccm_to_kg_s(G_sccm)
        v      = poiseuille_velocity(r, G_kg_s, R_TUBE, P_PA, T_NEUTRAL)
        v0_val = v[0]
        ax.plot(r * 1e3, v / v0_val,
                color=color, ls="--",
                label=f"$v(r)/v_0$,  G = {G_sccm:.0f} sccm")

    ax.set_xlabel("r, мм")
    ax.set_ylabel("Нормированная величина")
    ax.set_title(
        "Профиль скорости газа и концентрации плазмы\n"
        f"p = {P_PA:.0f} Па,  R = {R_TUBE*1e3:.0f} мм,  "
        f"$H_{{wall}}$ = {H_WALL:.0e} А/м"
    )
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, R_TUBE * 1e3)
    ax.set_ylim(0, 1.05)

    # Стрелки-аннотации
    ax.annotate(
        "плазма\n(скин-слой)",
        xy=(R_TUBE * 1e3 * 0.85, 0.45),
        xytext=(R_TUBE * 1e3 * 0.60, 0.65),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, color="black",
    )
    fig.tight_layout()
    _savefig(fig, save_path)


# ── Рис. П3: история бисекции (log n_e0 vs λ₀) ───────────────────────────────

def plot_lambda_scan(save_path):
    """
    Рис. П4 — λ₀(n_e0) для всех 9 пар (p, R).

    Каждая кривая показывает, при каком n_e0 достигается баланс λ₀=1.
    Если кривая пересекает пунктир λ₀=1 — разряд самосогласован при данном H_wall.
    Если не пересекает — разряд либо «перегрет» (λ₀>1 всюду), либо
    не поджигается (λ₀<1 всюду) при данном H_wall.
    """
    p_values = [66.5, 133.0, 266.0]
    R_values = [0.008, 0.012, 0.018]
    colors   = ["#1565C0", "#C62828", "#2E7D32"]   # синий, красный, зелёный
    ls_list  = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle(
        r"Зависимость $\lambda_0$ от $n_{e0}$ при разных параметрах разряда"
        f"\n$H_{{wall}}$ = {H_WALL:.0e} А/м,  аргон,  1.76 МГц",
        fontsize=12,
    )

    for ax, p_pa in zip(axes, p_values):
        for R, color, ls in zip(R_values, colors, ls_list):
            print(f"    scan p={p_pa:.0f} Pa, R={R*1e3:.0f} mm ...",
                  end=" ", flush=True)
            ne0_arr, lam0_arr = scan_lambda_ne0(p_pa, R, H_WALL, n_pts=20)
            ne0_star, _ = _find_crossing(ne0_arr, lam0_arr)

            label = f"R = {R*1e3:.0f} мм"
            ax.semilogx(ne0_arr, lam0_arr, color=color, ls=ls,
                        lw=1.8, label=label)

            # Отметить точку пересечения
            if ne0_star is not None:
                ax.axvline(ne0_star, color=color, ls=ls, lw=0.8, alpha=0.5)
                ax.plot(ne0_star, 1.0, "o", color=color, ms=7, zorder=5)
            print("done")

        ax.axhline(1.0, color="black", ls="-", lw=1.5,
                   label=r"$\lambda_0 = 1$ (баланс)")
        ax.set_xlabel(r"$n_{e0}$, м$^{-3}$")
        ax.set_ylabel(r"$\lambda_0$")
        ax.set_title(f"p = {p_pa:.1f} Па  ({p_pa/133.322:.2f} Торр)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, min(ax.get_ylim()[1], 10))  # ограничить ось Y

    fig.tight_layout()
    _savefig(fig, save_path)


def plot_bisection_history(save_path):
    """
    Показывает, как бисекция сходится к λ₀ = 1 при поиске n_e0*.
    Полезно для раздела «Численный метод».
    """
    print("  Бисекция для истории сходимости ...", flush=True)
    res = find_n_e0(
        N=N_GRID, R=R_TUBE, p_pa=P_PA, H_wall=H_WALL,
        tol_lambda=0.005, max_bisect=50, verbose=False,
    )
    history = res["history"]
    ne_vals  = [h[0] for h in history]
    lam_vals = [h[1] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(r"Сходимость бисекции по $n_{e0}$", fontsize=13)

    ax_lam, ax_err = axes

    steps = np.arange(1, len(history) + 1)

    ax_lam.plot(steps, lam_vals, "o-", color="#1565C0", ms=5)
    ax_lam.axhline(1.0, color="red", ls="--", lw=1.2, label=r"$\lambda_0 = 1$")
    ax_lam.set_xlabel("Шаг бисекции")
    ax_lam.set_ylabel(r"$\lambda_0$")
    ax_lam.set_title(r"Значение $\lambda_0$ по шагам")
    ax_lam.legend()
    ax_lam.grid(True, alpha=0.3)

    err_vals = np.abs(np.array(lam_vals) - 1.0)
    ax_err.semilogy(steps, err_vals, "s-", color="#C62828", ms=5)
    ax_err.set_xlabel("Шаг бисекции")
    ax_err.set_ylabel(r"$|\lambda_0 - 1|$")
    ax_err.set_title("Невязка бисекции (лог. масштаб)")
    ax_err.grid(True, alpha=0.3, which="both")

    # Аннотация финального значения
    ax_lam.annotate(
        f"n_e0* = {res['n_e0']:.2e} м⁻³\nλ₀ = {res['lambda0']:.4f}",
        xy=(len(history), lam_vals[-1]),
        xytext=(max(1, len(history) - 4), lam_vals[-1] + 0.15),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )

    fig.tight_layout()
    _savefig(fig, save_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main(save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        def path(name): return os.path.join(save_dir, name)
    else:
        def path(_): return None

    # ── Таблицы ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ТАБЛИЦЫ ДЛЯ ДИПЛОМНОЙ РАБОТЫ")
    print("="*60)

    print("\n[1/3] Таблица 1: параметрический анализ (p × R)...")
    table_parametric(save_dir)

    print("[2/3] Таблица 2: влияние включения...")
    table_inclusion(save_dir)

    print("[3/3] Таблица 3: профиль Пуазейля...")
    table_poiseuille(save_dir)

    # ── Рисунки ───────────────────────────────────────────────────────────────
    print("="*60)
    print("  РИСУНКИ ДЛЯ ДИПЛОМНОЙ РАБОТЫ")
    print("="*60)

    print("\n[Рис. П1] Профили скорости Пуазейля v(r)...")
    plot_poiseuille_profiles(path("figP1_poiseuille_profiles.png"))

    print("[Рис. П2] Профиль скорости vs профиль плазмы...")
    plot_velocity_vs_density(path("figP2_velocity_vs_density.png"))

    print("[Рис. П3] История сходимости бисекции...")
    plot_bisection_history(path("figP3_bisection_convergence.png"))

    print("[Рис. П4] lambda_0(n_e0) для разных p и R ...")
    plot_lambda_scan(path("figP4_lambda_scan.png"))

    print("\n" + "="*60)
    print("  Gotovo.")
    if save_dir:
        print(f"  Fajly sohraneny v: {os.path.abspath(save_dir)}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация таблиц и графиков для дипломной работы"
    )
    parser.add_argument(
        "--save", dest="save_dir",
        nargs="?", const="plots",
        default=None,
        metavar="DIR",
        help="Каталог для сохранения (по умолчанию plots/)",
    )
    args = parser.parse_args()
    main(save_dir=args.save_dir)
