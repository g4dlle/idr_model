"""
run_inclusion.py -- demonstration script for the IDR model with conductive inclusion.

Runs the solver for several inclusion radii and produces comparative figures:

  Figure 1 -- Dashboard for the case with inclusion (r_inc = 4 mm)
  Figure 2 -- Comparison: sigma/sigma_0 for different inclusion radii
  Figure 3 -- Field profiles comparison (H, E, sigma) with/without inclusion
  Figure 4 -- Joule dissipation Q(r) with inclusion

Usage
-----
  cd idr_model
  python run_inclusion.py              # show figures interactively
  python run_inclusion.py --save       # save PNG files to plots/
"""

import sys
import os
import argparse
import numpy as np

# Allow running from the idr_model/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from solver import solve_idr
from postprocess import (
    summary,
    plot_dashboard,
    plot_sigma_norm,
    plot_joule,
    plot_fields_comparison,
    joule_dissipation,
    total_power,
)
from config import P_PA, R_TUBE, H_WALL


def _calc_metrics(result):
    """Return key scalar metrics for inclusion/no-inclusion comparison."""
    r = result["r"]
    u = result["u"]
    v = result["v"]
    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]
    sigma_mod2 = sigma_a**2 + sigma_p**2
    q = joule_dissipation(r, sigma_a, sigma_mod2, u, v)
    p_total = total_power(r, q)
    return {
        "u0": float(u[0]),
        "vmax": float(np.max(v)),
        "sigma_max": float(np.max(sigma_a)),
        "power": float(p_total),
    }


def _pct_change(new, old):
    if abs(old) < 1e-300:
        return np.nan
    return 100.0 * (new / old - 1.0)


def print_inclusion_delta_table(results, r_inc_values):
    """Print compact comparison table relative to r_inc = 0 case."""
    if not results:
        return

    base = _calc_metrics(results[0])
    print("\nComparison vs no inclusion (r_inc = 0 mm):")
    print("-" * 92)
    print(f"{'r_inc, mm':>9} | {'vmax':>12} | {'dvmax,%':>8} | "
          f"{'sigma_max':>12} | {'dsigma,%':>8} | {'P, W/m':>12} | {'dP,%':>8}")
    print("-" * 92)

    for ri, res in zip(r_inc_values, results):
        m = _calc_metrics(res)
        dv = _pct_change(m["vmax"], base["vmax"])
        ds = _pct_change(m["sigma_max"], base["sigma_max"])
        dp = _pct_change(m["power"], base["power"])
        print(f"{ri*1e3:9.1f} | {m['vmax']:12.4e} | {dv:8.2f} | "
              f"{m['sigma_max']:12.4e} | {ds:8.2f} | {m['power']:12.4e} | {dp:8.2f}")
    print("-" * 92)


def main(save_dir=None):
    # ── Create output directory ──────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        def path(name): return os.path.join(save_dir, name)
    else:
        def path(_): return None   # show interactively

    # ── Figure 1: Dashboard with inclusion (r_inc = 4 mm) ────────────────────
    r_inc_main = 0.004    # m
    print(f"Running solver with inclusion r_inc = {r_inc_main*1e3:.0f} mm ...")
    result_inc = solve_idr(
        N=100,
        R=R_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        r_inc=r_inc_main,
        max_iter=500,
        tol=1e-6,
        relax=0.5,
        verbose=False,
    )
    summary(result_inc, P_PA)

    print("\nFigure 1: Dashboard (with inclusion)...")
    plot_dashboard(
        result_inc, p_pa=P_PA,
        title=(f"1D IDR model (Ar)  |  p = {P_PA:.0f} Pa,  "
               f"R = {R_TUBE*1e3:.1f} mm,  r_inc = {r_inc_main*1e3:.0f} mm"),
        save=path("fig1_inclusion_dashboard.png"),
    )

    # ── Figure 2: sigma/sigma_0 for several inclusion radii ──────────────────
    print("Figure 2: sigma/sigma_0 for different inclusion radii...")
    r_inc_values = [0.0, 0.002, 0.004, 0.006]    # m
    results_inc = []
    labels_inc = []
    for ri in r_inc_values:
        lbl = f"r_inc = {ri*1e3:.0f} mm" if ri > 0 else "без включения"
        print(f"  Solving: {lbl}...")
        res = solve_idr(
            N=100, R=R_TUBE, p_pa=P_PA, H_wall=H_WALL,
            r_inc=ri, max_iter=500, tol=1e-6, relax=0.5,
        )
        results_inc.append(res)
        labels_inc.append(lbl)
        print(f"    converged: {res['converged']}, iters: {res['n_iter']}")

    # Use one common sigma_0 from the reference case without inclusion.
    sigma0_ref = float(results_inc[0]["sigma_a"][0])
    if sigma0_ref <= 0.0:
        sigma0_ref = float(np.max(results_inc[0]["sigma_a"][:-1]))
    if sigma0_ref <= 0.0:
        sigma0_ref = 1.0
    sigma0_list = [sigma0_ref] * len(results_inc)

    print_inclusion_delta_table(results_inc, r_inc_values)

    plot_sigma_norm(
        results_inc,
        labels=labels_inc,
        sigma_0_list=sigma0_list,
        add_bessel=False,
        title=(r"Нормированная проводимость  $\sigma / \sigma_0$"
               f"  (p = {P_PA:.0f} Па)"),
        save=path("fig2_inclusion_sigma_norm.png"),
    )

    # ── Figure 3: Field profiles comparison with/without inclusion ────────────
    print("Figure 3: Field profiles with/without inclusion...")
    plot_fields_comparison(
        [results_inc[0], results_inc[2]],    # без включения vs r_inc=4 mm
        labels=["без включения", "r_inc = 4 mm"],
        title=f"Поля: эффект включения  (p = {P_PA:.0f} Па)",
        save=path("fig3_inclusion_fields.png"),
    )

    # ── Figure 4: Joule dissipation with inclusion ────────────────────────────
    print("Figure 4: Joule dissipation (r_inc = 4 mm)...")
    plot_joule(
        result_inc, p_pa=P_PA,
        save=path("fig4_inclusion_joule.png"),
    )

    if save_dir:
        print(f"\nAll figures saved to: {os.path.abspath(save_dir)}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 1D IDR model with conductive inclusion and plot results"
    )
    parser.add_argument(
        "--save", nargs="?", const="plots", metavar="DIR",
        help="Save figures to DIR (default: plots/)"
    )
    args = parser.parse_args()
    main(save_dir=args.save)
