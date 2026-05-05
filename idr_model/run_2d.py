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
import csv

# Saving figures must work on headless/broken-Tk Windows installations too.
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.dirname(__file__))

from solver_2d import solve_idr_2d
from solver import solve_idr
from self_consistent import solve_self_consistent
from self_consistent_2d import solve_self_consistent_2d
from postprocess_2d import (
    plot_dashboard_2d,
    plot_midplane,
    plot_midplane_detailed,
    plot_convergence_2d,
    compare_midplane_1d_2d,
    print_midplane_comparison,
    save_midplane_comparison_csv,
)
from config import P_PA, R_TUBE, H_WALL, L_TUBE, N_Z, N_GRID


def _resolve_save_dir(save_dir):
    if save_dir is None:
        return None
    if os.path.isabs(save_dir):
        return save_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, save_dir)


def main(save_dir=None, self_consistent=True):
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

    print("Figure 4: Detailed central-section comparison and mismatches...")
    mid_metrics = compare_midplane_1d_2d(result_2d, result_1d)
    print_midplane_comparison(mid_metrics)
    save_midplane_comparison_csv(mid_metrics, path("table_2d_midplane_mismatch.csv"))
    plot_midplane_detailed(result_2d, result_1d, metrics=mid_metrics,
                           save=path("fig_2d_midplane_detailed.png"))

    if not self_consistent:
        if output_dir:
            print(f"\nAll figures saved to: {os.path.abspath(output_dir)}/")
        else:
            print("\nDone.")
        return

    # Self-consistent comparison: n_e0* and H(r) in the central section.
    # This is intentionally coarser than production plots, because each point
    # runs a nested Maxwell solver inside bisection.
    print("\nRunning self-consistent 1D and 2D comparison for n_e0* and H(r)...")
    sc_1d = solve_self_consistent(
        N=60,
        R=R_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        tol_lambda=0.02,
        max_bisect=30,
        verbose=False,
        max_iter=500,
        tol=1e-5,
        relax=0.5,
    )
    sc_2d = solve_self_consistent_2d(
        Nr=60,
        Nz=N_Z,
        R=R_TUBE,
        L=L_TUBE,
        p_pa=P_PA,
        H_wall=H_WALL,
        bc_z_sigma="dirichlet",
        tol_lambda=0.02,
        max_bisect=30,
        verbose=False,
        max_iter=300,
        tol=1e-5,
        relax=0.5,
    )

    if sc_1d["solution"] is not None and sc_2d["solution"] is not None:
        sc_metrics = compare_midplane_1d_2d(sc_2d["solution"], sc_1d["solution"])
        ne0_1d = float(sc_1d["n_e0"])
        ne0_2d = float(sc_2d["n_e0"])
        ne0_rel = abs(ne0_2d - ne0_1d) / max(abs(ne0_1d), 1e-300)
        print("\nSelf-consistent mismatch:")
        print("-" * 72)
        print(f"  n_e0* 1D:        {ne0_1d:.4e} m^-3")
        print(f"  n_e0* 2D:        {ne0_2d:.4e} m^-3")
        print(f"  n_e0 mismatch:   {100*ne0_rel:.2f}%")
        print(f"  lambda0 1D/2D:   {sc_1d['lambda0']:.4f} / {sc_2d['lambda0']:.4f}")
        print(f"  converged 1D/2D: {sc_1d['converged']} / {sc_2d['converged']}")
        print(f"  H(r) max mismatch at z=L/2: {100*sc_metrics['H']['rel_max']:.2f}%")
        print(f"  H(r) L2 mismatch at z=L/2:  {100*sc_metrics['H']['rel_l2']:.2f}%")
        if not (sc_1d["converged"] and sc_2d["converged"]):
            print("  NOTE: n_e0 values are nearest returned bounds, not a lambda0=1 root.")
        print("-" * 72)

        if output_dir:
            sc_csv = path("table_2d_self_consistent_mismatch.csv")
            with open(sc_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["n_e0_1d_m^-3", ne0_1d])
                writer.writerow(["n_e0_2d_m^-3", ne0_2d])
                writer.writerow(["n_e0_rel_mismatch", ne0_rel])
                writer.writerow(["H_rel_max_midplane", sc_metrics["H"]["rel_max"]])
                writer.writerow(["H_rel_l2_midplane", sc_metrics["H"]["rel_l2"]])
                writer.writerow(["lambda0_1d", sc_1d["lambda0"]])
                writer.writerow(["lambda0_2d", sc_2d["lambda0"]])
                writer.writerow(["converged_1d", sc_1d["converged"]])
                writer.writerow(["converged_2d", sc_2d["converged"]])
            print(f"  [csv] saved: {sc_csv}")

        print("Figure 5: Self-consistent central-section comparison...")
        plot_midplane_detailed(sc_2d["solution"], sc_1d["solution"],
                               metrics=sc_metrics,
                               save=path("fig_2d_self_consistent_midplane.png"))
    else:
        print("  Self-consistent comparison skipped: one of the solutions is missing.")

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
    parser.add_argument(
        "--skip-self-consistent", action="store_true",
        help="Skip the expensive self-consistent 1D/2D comparison"
    )
    args = parser.parse_args()
    main(save_dir=args.save, self_consistent=not args.skip_self_consistent)
