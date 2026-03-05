"""
run_2d.py — demonstration script for the 2D axisymmetric IDR model.

Runs solve_idr_2d and produces contour plots and midplane comparisons.

Usage
-----
  cd idr_model
  python run_2d.py              # show figures
  python run_2d.py --save       # save PNGs to plots/
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from solver_2d import solve_idr_2d
from solver import solve_idr
from postprocess_2d import (
    plot_dashboard_2d,
    plot_midplane,
    plot_convergence_2d,
)
from config import P_PA, R_TUBE, H_WALL, L_TUBE, N_Z, N_GRID


def _resolve_save_dir(save_dir):
    if save_dir is None:
        return None
    if os.path.isabs(save_dir):
        return save_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, save_dir)


def main(save_dir=None):
    output_dir = _resolve_save_dir(save_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        def path(name): return os.path.join(output_dir, name)
    else:
        def path(_): return None

    # ── 2D solve ──────────────────────────────────────────────────────────
    print("Running 2D solver (default parameters)...")
    result_2d = solve_idr_2d(
        Nr=60,
        Nz=N_Z,
        R=R_TUBE,
        L=L_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        max_iter=300,
        tol=1e-5,
        relax=0.5,
        bc_z_sigma="dirichlet",
        verbose=True,
    )

    print(f"\n  Converged: {result_2d['converged']}")
    print(f"  Iterations: {result_2d['n_iter']}")

    # ── Dashboard ─────────────────────────────────────────────────────────
    print("\nFigure 1: 2D Dashboard...")
    plot_dashboard_2d(result_2d, p_pa=P_PA,
                      save=path("fig_2d_dashboard.png"))

    # ── Convergence ───────────────────────────────────────────────────────
    print("Figure 2: Convergence...")
    plot_convergence_2d(result_2d, tol=1e-5,
                        save=path("fig_2d_convergence.png"))

    # ── Midplane vs 1D ────────────────────────────────────────────────────
    print("Running 1D solver for comparison...")
    result_1d = solve_idr(
        N=60,
        R=R_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        max_iter=500,
        tol=1e-6,
        relax=0.5,
    )
    print(f"  1D converged: {result_1d['converged']}")

    print("Figure 3: Midplane comparison (2D vs 1D)...")
    plot_midplane(result_2d, result_1d=result_1d,
                  save=path("fig_2d_midplane.png"))

    if output_dir:
        print(f"\nAll figures saved to: {os.path.abspath(output_dir)}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 2D IDR model and plot results"
    )
    parser.add_argument(
        "--save", nargs="?", const="plots", metavar="DIR",
        help="Save figures to DIR (default: plots/)"
    )
    args = parser.parse_args()
    main(save_dir=args.save)
