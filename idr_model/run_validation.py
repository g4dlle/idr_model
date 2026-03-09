"""
run_validation.py — валидация модели ИДР по данным статьи 2025.

Запускает солвер при давлениях 1000, 5000, 10000 Па с транспортными
коэффициентами из BOLSIG+ (если есть) или аналитическими формулами,
и сравнивает ne с экспериментальными данными.

Usage
-----
  python run_validation.py                # запуск с аналитическими коэффициентами
  python run_validation.py --bolsig       # с данными BOLSIG+ из bolsig_data/
  python run_validation.py --save         # сохранить графики
  python run_validation.py --beta 1e-13   # с объёмной рекомбинацией
"""

import sys
import os
import argparse
import csv

import numpy as np

# Allow running from the idr_model/ directory
sys.path.insert(0, os.path.dirname(__file__))

from solver import solve_idr
from physics import collision_freq

MU_0 = 4.0 * np.pi * 1e-7
K_BOLTZMANN = 1.380649e-23
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31

# ── Параметры из validation_table.md ─────────────────────────────────────────
FREQUENCY_HZ = 1.76e6
OMEGA = 2 * np.pi * FREQUENCY_HZ
JET_DIAMETER = 1.5e-3        # м
JET_RADIUS = JET_DIAMETER / 2  # 0.75 мм
JET_LENGTH = 40e-3            # м (не используется в 1D)

# Экспериментальные данные
VALIDATION_PRESSURES = [1000.0, 5000.0, 10000.0]  # Па


def load_validation_csv(filepath: str) -> list[dict]:
    """Загружает validation_targets_by_pressure.csv."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def estimate_H_wall(p_pa: float, j_exp_kA_m2: float) -> float:
    """
    Оценка H_wall из экспериментальной плотности тока.

    В цилиндрическом ИДР связь H на стенке с плотностью тока:
      j ≈ σ_a · E  и  E ≈ ω·μ₀·H_wall·R/2
    Для грубой оценки: H_wall ≈ 2·j / (ω·μ₀·R · σ/σ_ratio)

    Альтернативно: используем закон Ампера для однородного тока
      H_wall = j · R / 2

    Parameters
    ----------
    p_pa        : давление [Па]
    j_exp_kA_m2 : плотность тока [кА/м²]

    Returns
    -------
    H_wall : А/м
    """
    j = j_exp_kA_m2 * 1e3  # кА/м² → А/м²
    # Из закона Ампера для цилиндра с однородным j:
    # ∮ H·dl = I_enc = j · π·R²  →  H_wall · 2π·R = j · π·R²
    # → H_wall = j · R / 2
    H_wall = j * JET_RADIUS / 2.0
    return H_wall


def run_single_pressure(p_pa: float, H_wall: float,
                        transport=None, beta_recomb=0.0,
                        verbose=False) -> dict:
    """Запуск солвера для одного давления."""
    result = solve_idr(
        N=100,
        R=JET_RADIUS,
        p_pa=p_pa,
        H_wall=H_wall,
        n_e0=1e16,
        max_iter=500,
        tol=1e-5,
        relax=0.3,
        transport=transport,
        beta_recomb=beta_recomb,
        verbose=verbose,
    )
    return result


def ne_max_from_result(result: dict, p_pa: float, nu_c_val=None) -> float:
    """Извлекает максимальную ne [м⁻³] из результата солвера."""
    sigma_a = result["sigma_a"]
    if nu_c_val is None:
        nu_c_val = collision_freq(p_pa)

    ne = sigma_a * M_ELECTRON * (nu_c_val**2 + OMEGA**2) / (E_CHARGE**2 * nu_c_val)
    return float(ne.max())


def run_validation(use_bolsig=False, beta_recomb=0.0, save_dir=None,
                   verbose=False):
    """
    Основная функция валидации.

    Parameters
    ----------
    use_bolsig   : использовать данные BOLSIG+ (если доступны)
    beta_recomb  : коэфф. рекомбинации [м³/с]
    save_dir     : директория для сохранения графиков (None → показать)
    verbose      : печатать итерации
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "validation_targets_by_pressure.csv")

    # Загрузка экспериментальных данных
    exp_data = load_validation_csv(csv_path)
    exp_by_p = {}
    for row in exp_data:
        p = float(row["pressure_pa"])
        exp_by_p[p] = row

    # Загрузка BOLSIG+ данных (если доступны)
    bolsig_dir = os.path.join(os.path.dirname(base_dir), "bolsigplus072024-win")
    bolsig_transports = {}

    if use_bolsig:
        try:
            from bolsig_parser import parse_bolsig_output
            from bolsig_transport import BolsigTransport

            for p_pa in VALIDATION_PRESSURES:
                data_file = os.path.join(bolsig_dir, f"bolsig_ar_{int(p_pa)}Pa.dat")
                if os.path.exists(data_file):
                    data = parse_bolsig_output(data_file)
                    bolsig_transports[p_pa] = BolsigTransport(data, p_pa=p_pa)
                    print(f"  [bolsig] Loaded transport for p = {p_pa} Pa")
                else:
                    print(f"  [bolsig] WARNING: {data_file} not found, "
                          f"using analytical formulas for p = {p_pa} Pa")
        except ImportError as e:
            print(f"  [bolsig] Import error: {e}")
            print(f"  [bolsig] Falling back to analytical formulas")

    # ── Запуск расчётов ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  IDR Model Validation")
    print("=" * 72)
    print(f"  Geometry: R = {JET_RADIUS*1e3:.2f} mm, f = {FREQUENCY_HZ/1e6:.2f} MHz")
    print(f"  Transport: {'BOLSIG+' if use_bolsig else 'analytical (Abdullin-Zheltukhin 1985)'}")
    print(f"  Recombination: β = {beta_recomb:.2e} m³/s")
    print("=" * 72)
    print()

    results = {}

    for p_pa in VALIDATION_PRESSURES:
        exp = exp_by_p.get(p_pa, {})
        j_exp = float(exp.get("j_kA_m2_exp", 837))
        ne_exp_cm3 = float(exp.get("ne_cm3_exp_Hb", 2.2e16))
        ne_exp_m3 = ne_exp_cm3 * 1e6  # см⁻³ → м⁻³
        regime = exp.get("regime_exp", "?")

        H_wall = estimate_H_wall(p_pa, j_exp)

        transport = bolsig_transports.get(p_pa, None)

        print(f"── p = {p_pa:.0f} Pa ({p_pa/133.322:.1f} Torr) ──")
        print(f"    H_wall = {H_wall:.1f} A/m  (from j_exp = {j_exp} kA/m²)")
        print(f"    regime: {regime}")

        result = run_single_pressure(
            p_pa=p_pa,
            H_wall=H_wall,
            transport=transport,
            beta_recomb=beta_recomb,
            verbose=verbose,
        )

        ne_model = ne_max_from_result(result, p_pa)
        results[p_pa] = {
            "result": result,
            "ne_model": ne_model,
            "ne_exp": ne_exp_m3,
            "H_wall": H_wall,
            "converged": result["converged"],
        }

        print(f"    converged: {result['converged']} ({result['n_iter']} iter)")
        print(f"    ne_model(0) = {ne_model:.3e} m⁻³  ({ne_model/1e6:.3e} cm⁻³)")
        print(f"    ne_exp       = {ne_exp_m3:.3e} m⁻³  ({ne_exp_cm3:.3e} cm⁻³)")
        if ne_exp_m3 > 0:
            ratio = ne_model / ne_exp_m3
            print(f"    ratio ne_model/ne_exp = {ratio:.3f}")
        print()

    # ── Сводная таблица ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  Summary Table")
    print("-" * 72)
    print(f"  {'p [Pa]':>10} | {'ne_model [cm⁻³]':>16} | {'ne_exp [cm⁻³]':>14} | "
          f"{'ratio':>8} | {'conv':>5} | {'regime'}")
    print("-" * 72)

    for p_pa in VALIDATION_PRESSURES:
        r = results[p_pa]
        ne_m_cm3 = r["ne_model"] / 1e6
        ne_e_cm3 = r["ne_exp"] / 1e6
        ratio = r["ne_model"] / r["ne_exp"] if r["ne_exp"] > 0 else float("inf")
        conv = "yes" if r["converged"] else "NO"
        exp = exp_by_p.get(p_pa, {})
        regime = exp.get("regime_exp", "?")

        print(f"  {p_pa:>10.0f} | {ne_m_cm3:>16.3e} | {ne_e_cm3:>14.3e} | "
              f"{ratio:>8.3f} | {conv:>5} | {regime}")

    print("=" * 72)

    # ── Графики ──────────────────────────────────────────────────────────────
    try:
        _plot_validation(results, VALIDATION_PRESSURES, exp_by_p,
                         beta_recomb=beta_recomb, use_bolsig=use_bolsig,
                         save_dir=save_dir)
    except Exception as e:
        print(f"\n  [plot] Error: {e}")

    return results


def _plot_validation(results, pressures, exp_by_p,
                     beta_recomb=0.0, use_bolsig=False,
                     save_dir=None):
    """Графики валидации: ne(p) и профили σ(r)."""
    import matplotlib.pyplot as plt

    # ── График 1: ne(p) модель vs эксперимент ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    p_arr = np.array(pressures)
    ne_model = np.array([results[p]["ne_model"] / 1e6 for p in pressures])  # cm⁻³
    ne_exp = np.array([results[p]["ne_exp"] / 1e6 for p in pressures])

    ax1.semilogy(p_arr / 1e3, ne_model, "bo-", ms=8, lw=2, label="Model", zorder=3)
    ax1.semilogy(p_arr / 1e3, ne_exp, "rs--", ms=10, lw=1.5, label="Experiment (Hβ)")
    ax1.set_xlabel("Pressure [kPa]", fontsize=12)
    ax1.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=12)
    transport_label = "BOLSIG+" if use_bolsig else "analytical"
    beta_label = f", β={beta_recomb:.0e}" if beta_recomb > 0 else ""
    ax1.set_title(f"Electron density vs pressure\n({transport_label}{beta_label})")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── График 2: профили σ(r) для разных давлений ───────────────────────────
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, p_pa in enumerate(pressures):
        r = results[p_pa]["result"]["r"]
        rn = r / r[-1]
        sa = results[p_pa]["result"]["sigma_a"]
        s0 = sa[0] if sa[0] > 0 else 1.0
        ax2.plot(rn, sa / s0, color=colors[i], lw=2,
                 label=f"p = {p_pa/1e3:.0f} kPa")

    ax2.set_xlabel("r / R", fontsize=12)
    ax2.set_ylabel(r"$\sigma_a / \sigma_a(0)$", fontsize=12)
    ax2.set_title("Normalized conductivity profiles")
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "validation_ne_vs_pressure.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"\n  [plot] Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Validate IDR model against experimental data"
    )
    parser.add_argument("--bolsig", action="store_true",
                        help="Use BOLSIG+ transport data (if available)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Volume recombination coefficient β [m³/s]")
    parser.add_argument("--save", nargs="?", const="plots", metavar="DIR",
                        help="Save figures to DIR (default: plots/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print solver iterations")
    args = parser.parse_args()

    save_dir = None
    if args.save:
        base = os.path.dirname(os.path.abspath(__file__))
        save_dir = args.save if os.path.isabs(args.save) else os.path.join(base, args.save)

    run_validation(
        use_bolsig=args.bolsig,
        beta_recomb=args.beta,
        save_dir=save_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
