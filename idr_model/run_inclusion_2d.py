"""
run_inclusion_2d.py -- demonstration script for the 2D IDR model with
conductive inclusion.

Runs the 2D solver for several inclusion radii and produces comparative
figures:

  Figure 1 -- Dashboard for the case with inclusion (r_inc = 4 mm)
  Figure 2 -- Comparison: sigma/sigma_0 at midplane for different r_inc
  Figure 3 -- Field profiles comparison (H, E, sigma) at midplane

Usage
-----
  cd idr_model
  python run_inclusion_2d.py              # show figures interactively
  python run_inclusion_2d.py --save       # save PNG files to plots/
"""

import sys
import os
import argparse
import numpy as np

# Allow running from the idr_model/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from solver_2d import solve_idr_2d
from postprocess_2d import (
    plot_dashboard_2d,
    plot_midplane_inclusion,
    plot_fields_comparison_2d,
)
from config import P_PA, R_TUBE, H_WALL, L_TUBE, N_Z


def _calc_metrics_2d(result):
    """Return key scalar metrics for a 2D result at midplane."""
    z = result["z"]
    j_mid = len(z) // 2
    sigma_a = result["sigma_a"]
    n_e = result["n_e"]
    return {
        "sigma_max": float(np.max(sigma_a[:, j_mid])),
        "n_e_max":   float(np.max(n_e[:, j_mid])),
        "sigma_max_global": float(np.max(sigma_a)),
        "n_e_max_global":   float(np.max(n_e)),
    }


def _pct_change(new, old):
    if abs(old) < 1e-300:
        return np.nan
    return 100.0 * (new / old - 1.0)


def print_inclusion_delta_table_2d(results, r_inc_values):
    """Print compact comparison table relative to r_inc = 0 case."""
    if not results:
        return

    base = _calc_metrics_2d(results[0])
    print("\n2D  Comparison vs no inclusion (r_inc = 0 mm), midplane z=L/2:")
    print("-" * 88)
    print(f"{'r_inc, mm':>9} | {'sigma_max':>12} | {'dsigma,%':>8} | "
          f"{'n_e_max':>12} | {'dn_e,%':>8} | "
          f"{'iters':>5} | {'convrgd':>7}")
    print("-" * 88)

    for ri, res in zip(r_inc_values, results):
        m = _calc_metrics_2d(res)
        ds = _pct_change(m["sigma_max"], base["sigma_max"])
        dn = _pct_change(m["n_e_max"], base["n_e_max"])
        cvg = "yes" if res["converged"] else "no"
        print(f"{ri*1e3:9.1f} | {m['sigma_max']:12.4e} | {ds:8.2f} | "
              f"{m['n_e_max']:12.4e} | {dn:8.2f} | "
              f"{res['n_iter']:5d} | {cvg:>7}")
    print("-" * 88)


def _resolve_save_dir(save_dir):
    if save_dir is None:
        return None
    if os.path.isabs(save_dir):
        return save_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, save_dir)


def main(save_dir=None):
    output_dir = _resolve_save_dir(save_dir)
    # ── Create output directory ────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        def path(name): return os.path.join(output_dir, name)
    else:
        def path(_): return None   # show interactively

    # ── Figure 1: 2D Dashboard with inclusion (r_inc = 4 mm) ──────────────
    r_inc_main = 0.004    # m
    print(f"Running 2D solver with inclusion r_inc = {r_inc_main*1e3:.0f} mm ...")
    result_inc = solve_idr_2d(
        Nr=60,
        Nz=N_Z,
        R=R_TUBE,
        L=L_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        r_inc=r_inc_main,
        max_iter=300,
        tol=1e-5,
        relax=0.5,
        bc_z_sigma="dirichlet",
        verbose=False,
    )
    print(f"  Converged: {result_inc['converged']}, "
          f"iters: {result_inc['n_iter']}")

    print("\nFigure 1: 2D Dashboard (with inclusion)...")
    plot_dashboard_2d(
        result_inc, p_pa=P_PA,
        title=(f"2D IDR model (Ar)  |  p = {P_PA:.0f} Pa,  "
               f"R = {R_TUBE*1e3:.1f} mm,  r_inc = {r_inc_main*1e3:.0f} mm"),
        save=path("fig_2d_inc_dashboard.png"),
    )

    # ── Figure 2: sigma/sigma_0 at midplane for several inclusion radii ───
    print("Figure 2: sigma/sigma_0 at midplane for different r_inc...")
    r_inc_values = [0.0, 0.002, 0.004, 0.006]    # m
    results_inc = []
    labels_inc = []
    for ri in r_inc_values:
        lbl = f"r_inc = {ri*1e3:.0f} мм" if ri > 0 else "без включения"
        print(f"  Solving 2D: {lbl}...")
        res = solve_idr_2d(
            Nr=60, Nz=N_Z, R=R_TUBE, L=L_TUBE,
            p_pa=P_PA, H_wall=H_WALL,
            r_inc=ri,
            max_iter=300, tol=1e-5, relax=0.5,
            bc_z_sigma="dirichlet",
        )
        results_inc.append(res)
        labels_inc.append(lbl)
        print(f"    converged: {res['converged']}, iters: {res['n_iter']}")

    # Common sigma_0 from the no-inclusion case at midplane
    z_ref = results_inc[0]["z"]
    j_mid_ref = len(z_ref) // 2
    sigma0_ref = float(results_inc[0]["sigma_a"][0, j_mid_ref])
    if sigma0_ref <= 0.0:
        sigma0_ref = float(np.max(results_inc[0]["sigma_a"][:, j_mid_ref]))
    if sigma0_ref <= 0.0:
        sigma0_ref = 1.0
    sigma0_list = [sigma0_ref] * len(results_inc)

    print_inclusion_delta_table_2d(results_inc, r_inc_values)

    plot_midplane_inclusion(
        results_inc,
        labels=labels_inc,
        sigma_0_list=sigma0_list,
        title=(r"Нормированная проводимость  $\sigma / \sigma_0$"
               f"  (2D, p = {P_PA:.0f} Па)"),
        save=path("fig_2d_inc_sigma_norm.png"),
    )

    # ── Figure 3: Field profiles comparison with/without inclusion ────────
    print("Figure 3: Field profiles (2D) with/without inclusion...")
    plot_fields_comparison_2d(
        [results_inc[0], results_inc[2]],    # без включения vs r_inc=4 mm
        labels=["без включения", "r_inc = 4 мм"],
        title=f"Поля: эффект включения (2D, p = {P_PA:.0f} Па)",
        save=path("fig_2d_inc_fields.png"),
    )

    if output_dir:
        print(f"\nAll figures saved to: {os.path.abspath(output_dir)}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 2D IDR model with conductive inclusion and plot results"
    )
    parser.add_argument(
        "--save", nargs="?", const="plots", metavar="DIR",
        help="Save figures to DIR (default: plots/)"
    )
    args = parser.parse_args()
    main(save_dir=args.save)
