"""
run_and_plot.py -- demonstration script for the 1D IDR model.

Runs the solver for default parameters and produces four figures:

  Figure 1 -- Dashboard (2x3): all field profiles + convergence
  Figure 2 -- Normalised conductivity sigma/sigma0  (Fig.1 of the paper style)
  Figure 3 -- Joule dissipation Q(r) with total power annotation
  Figure 4 -- Parametric sweep: sigma/sigma0 for three different tube radii

Usage
-----
  cd idr_model
  python run_and_plot.py              # show figures interactively
  python run_and_plot.py --save       # save PNG files to plots/
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
    plot_convergence,
    plot_parametric,
    plot_fields_comparison,
)
from config import P_PA, R_TUBE, H_WALL, N_GRID


def main(save_dir=None):
    # ── Create output directory ──────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        def path(name): return os.path.join(save_dir, name)
    else:
        def path(_): return None   # show interactively

    # ── Figure 1: Dashboard --------------------------------------------------
    print("Running solver (default parameters)...")
    result = solve_idr(
        N=100,
        R=R_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        max_iter=500,
        tol=1e-6,
        relax=0.5,
        verbose=False,
    )
    summary(result, P_PA)

    print("\nFigure 1: Dashboard...")
    plot_dashboard(
        result, p_pa=P_PA,
        save=path("fig1_dashboard.png")
    )

    # ── Figure 2: sigma/sigma0 profile with J0 overlay ──────────────────────
    print("Figure 2: sigma/sigma_0 profile...")
    plot_sigma_norm(
        result,
        labels=[f"p = {P_PA:.0f} Pa,  R = {R_TUBE*1e3:.0f} mm"],
        add_bessel=True,
        save=path("fig2_sigma_norm.png"),
    )

    # ── Figure 3: Joule dissipation Q(r) ────────────────────────────────────
    print("Figure 3: Joule dissipation...")
    plot_joule(result, p_pa=P_PA,
               save=path("fig3_joule.png"))

    # ── Figure 4: Parametric sweep over tube radius --------------------------
    print("Figure 4: Parametric sweep over R...")
    R_values = [0.008, 0.012, 0.018]   # m
    results_R = []
    for R_val in R_values:
        res = solve_idr(
            N=100,
            R=R_val,
            p_pa=P_PA,
            H_wall=H_WALL,
            max_iter=500,
            tol=1e-6,
            relax=0.5,
            verbose=False,
        )
        results_R.append(res)
        print(f"  R = {R_val*1e3:.0f} mm  |  converged: {res['converged']}"
              f"  |  iters: {res['n_iter']}")

    plot_parametric(
        param_values=[f"{R*1e3:.0f}" for R in R_values],
        results=results_R,
        param_name="R",
        param_unit="mm",
        field="sigma_norm",
        title=r"Effect of tube radius on $\sigma / \sigma_0$"
              f"   (p = {P_PA:.0f} Pa)",
        save=path("fig4_parametric_R.png"),
    )

    # ── Figure 5: Field profiles for several radii (comparison) ─────────────
    print("Figure 5: Field profiles comparison (H, E, sigma)...")
    plot_fields_comparison(
        results_R,
        labels=[f"R = {R*1e3:.0f} mm" for R in R_values],
        title=f"Field profiles for different radii  (p = {P_PA:.0f} Pa)",
        save=path("fig5_fields_comparison.png"),
    )

    # ── Figure 6: Parametric sweep over pressure ─────────────────────────────
    print("Figure 6: Parametric sweep over pressure...")
    p_values_pa = [66.5, 133.0, 266.0]   # Pa (0.5, 1, 2 Torr)
    results_p = []
    for p_val in p_values_pa:
        res = solve_idr(
            N=100,
            R=R_TUBE,
            p_pa=p_val,
            H_wall=H_WALL,
            max_iter=500,
            tol=1e-6,
            relax=0.5,
            verbose=False,
        )
        results_p.append(res)
        print(f"  p = {p_val:.1f} Pa  |  converged: {res['converged']}"
              f"  |  iters: {res['n_iter']}")

    plot_parametric(
        param_values=[f"{p:.0f}" for p in p_values_pa],
        results=results_p,
        param_name="p",
        param_unit="Pa",
        field="sigma_norm",
        title=r"Effect of pressure on $\sigma / \sigma_0$"
              f"   (R = {R_TUBE*1e3:.0f} mm)",
        save=path("fig6_parametric_p.png"),
    )

    # ── Figure 7: Convergence comparison for two relaxation factors ──────────
    print("Figure 7: Convergence comparison (relax 0.3 vs 0.7)...")
    res03 = solve_idr(N=100, p_pa=P_PA, max_iter=500, tol=1e-8, relax=0.3)
    res07 = solve_idr(N=100, p_pa=P_PA, max_iter=500, tol=1e-8, relax=0.7)

    plot_convergence(
        [res03, res07],
        labels=["relax = 0.3", "relax = 0.7"],
        tol=1e-8,
        save=path("fig7_convergence.png"),
    )

    if save_dir:
        print(f"\nAll figures saved to: {os.path.abspath(save_dir)}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 1D IDR model and plot results")
    parser.add_argument(
        "--save", nargs="?", const="plots", metavar="DIR",
        help="Save figures to DIR (default: plots/)"
    )
    args = parser.parse_args()
    main(save_dir=args.save)
