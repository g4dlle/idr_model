"""
postprocess_2d.py -- visualization for the 2D axisymmetric IDR model.

Provides contour plots (r, z) for all fields and midplane comparison with 1D.
"""

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_STYLE = {
    "figure.dpi":        120,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "axes.grid":         False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _apply_style():
    if HAS_MPL:
        plt.rcParams.update(_STYLE)


def _save_or_show(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
        print(f"  [plot] saved: {path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Single field contour
# ---------------------------------------------------------------------------

def plot_field_2d(result, field="u", title=None, save=None):
    """
    Contour plot of a 2D field on (r, z) axes.

    Parameters
    ----------
    result : dict from solve_idr_2d
    field  : "u", "v", "sigma_a", "n_e"
    title  : figure title
    save   : file path, None → show
    """
    if not HAS_MPL:
        return
    _apply_style()

    r = result["r"]
    z = result["z"]
    data = result[field]
    R, Z = np.meshgrid(z, r)  # note: meshgrid(z, r) → shape (Nr+1, Nz+1)

    _labels = {
        "u":       (r"$|H|^2$  [A$^2$/m$^2$]", r"$|H|^2(r,z)$"),
        "v":       (r"$|E|^2$  [V$^2$/m$^2$]", r"$|E|^2(r,z)$"),
        "sigma_a": (r"$\sigma_a$  [S/m]",       r"$\sigma_a(r,z)$"),
        "n_e":     (r"$n_e$  [m$^{-3}$]",       r"$n_e(r,z)$"),
    }
    cbar_label, default_title = _labels.get(field, (field, field))

    fig, ax = plt.subplots(figsize=(10, 4))
    cf = ax.contourf(Z * 1e3, R * 1e3, data, levels=30, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax, label=cbar_label)
    ax.set_xlabel("z  [mm]")
    ax.set_ylabel("r  [mm]")
    ax.set_title(title or default_title)
    ax.set_aspect("auto")
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Dashboard 2×2
# ---------------------------------------------------------------------------

def plot_dashboard_2d(result, p_pa=None, title=None, save=None):
    """
    2×2 contour dashboard: |H|², |E|², σ_a, n_e.
    """
    if not HAS_MPL:
        return
    _apply_style()

    r = result["r"]
    z = result["z"]
    R_m, Z_m = np.meshgrid(z, r)
    z_mm = Z_m * 1e3
    r_mm = R_m * 1e3

    fields = [
        ("u",       r"$|H|^2$",      "plasma"),
        ("v",       r"$|E|^2$",      "inferno"),
        ("sigma_a", r"$\sigma_a$",   "viridis"),
        ("n_e",     r"$n_e$",        "magma"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, (key, label, cmap) in zip(axes, fields):
        data = result[key]
        cf = ax.contourf(z_mm, r_mm, data, levels=30, cmap=cmap)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("r  [mm]")
        ax.set_title(f"{label}(r, z)")
        ax.set_aspect("auto")

    if title is None and p_pa is not None:
        R_val = r[-1]
        L_val = z[-1]
        title = (f"2D IDR model (Ar)  |  "
                 f"p = {p_pa:.0f} Pa,  R = {R_val*1e3:.1f} mm,"
                 f"  L = {L_val*1e3:.1f} mm")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Midplane profiles (z = L/2) — comparison with optional 1D result
# ---------------------------------------------------------------------------

def plot_midplane(result_2d, result_1d=None, save=None):
    """
    Radial profiles at z = L/2.  Optionally overlay 1D solution.
    """
    if not HAS_MPL:
        return
    _apply_style()

    r = result_2d["r"]
    z = result_2d["z"]
    j_mid = len(z) // 2

    R = r[-1]
    rn = r / R

    fields = [
        ("u",       r"$|H|^2$  [A$^2$/m$^2$]"),
        ("v",       r"$|E|^2$  [V$^2$/m$^2$]"),
        ("sigma_a", r"$\sigma_a$  [S/m]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Midplane profiles  (z = {z[j_mid]*1e3:.1f} mm)",
                 fontsize=13, fontweight="bold")

    for ax, (key, ylabel) in zip(axes, fields):
        ax.plot(rn, result_2d[key][:, j_mid], "b-", lw=2, label="2D")
        if result_1d is not None:
            r1 = result_1d["r"]
            ax.plot(r1 / r1[-1], result_1d[key], "r--", lw=1.5, label="1D")
        ax.set_xlabel("r / R")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.legend()

    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def plot_convergence_2d(result, tol=None, save=None):
    """Semilog convergence history."""
    if not HAS_MPL:
        return
    _apply_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(result["residuals"], "b-", lw=2)
    if tol is not None:
        ax.axhline(tol, color="black", lw=1.2, ls="--", label=f"tol = {tol:.0e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title("2D IDR convergence")
    ok = "converged" if result["converged"] else "NOT converged"
    ax.text(0.97, 0.97, ok, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#2ca02c" if result["converged"] else "#d62728")
    if tol:
        ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Midplane sigma/sigma_0 for several inclusion radii
# ---------------------------------------------------------------------------

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_midplane_inclusion(results, labels=None,
                            sigma_0_list=None,
                            title=None, save=None):
    """
    Plot sigma_a / sigma_0 at z = L/2 for several 2D results
    (typically different inclusion radii).

    Parameters
    ----------
    results      : list of dicts from solve_idr_2d
    labels       : curve labels
    sigma_0_list : reference sigma_0 per curve (None → use midplane max)
    title        : figure title
    save         : file path, None → show
    """
    if not HAS_MPL:
        return
    _apply_style()

    if isinstance(results, dict):
        results = [results]
    if labels is None:
        labels = [f"curve {i+1}" for i in range(len(results))]
    if sigma_0_list is None:
        sigma_0_list = [None] * len(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (res, lbl, s0) in enumerate(zip(results, labels, sigma_0_list)):
        r = res["r"]
        z = res["z"]
        R = r[-1]
        rn = r / R
        j_mid = len(z) // 2
        sa_mid = res["sigma_a"][:, j_mid]

        if s0 is None or s0 == 0.0:
            s0 = sa_mid[0] if sa_mid[0] > 0 else float(np.max(sa_mid))
        if s0 <= 0.0:
            s0 = 1.0

        ax.plot(rn, sa_mid / s0,
                color=_COLORS[i % len(_COLORS)],
                lw=2, label=lbl, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("r / R", fontsize=12)
    ax.set_ylabel(r"$\sigma_a / \sigma_0$", fontsize=12)
    if title is None:
        title = r"Нормированная проводимость $\sigma / \sigma_0$  (2D, midplane)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Side-by-side field comparison at midplane
# ---------------------------------------------------------------------------

def plot_fields_comparison_2d(results, labels=None,
                               title="Поля на средней плоскости (2D)",
                               save=None):
    """
    Three subplots: |H|², |E|², sigma_a at z = L/2 for several 2D solutions.

    Parameters
    ----------
    results : list of dicts from solve_idr_2d
    labels  : curve labels
    title   : figure title
    save    : file path, None → show
    """
    if not HAS_MPL:
        return
    _apply_style()

    if isinstance(results, dict):
        results = [results]
    if labels is None:
        labels = [f"#{i+1}" for i in range(len(results))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (res, lbl) in enumerate(zip(results, labels)):
        r = res["r"]
        z = res["z"]
        rn = r / r[-1]
        j_mid = len(z) // 2
        c = _COLORS[i % len(_COLORS)]

        axes[0].plot(rn, res["u"][:, j_mid],       color=c, lw=2, label=lbl)
        axes[1].plot(rn, res["v"][:, j_mid],       color=c, lw=2, label=lbl)
        axes[2].plot(rn, res["sigma_a"][:, j_mid], color=c, lw=2, label=lbl)

    ylabels = [r"$|H|^2$  [A$^2$/m$^2$]",
               r"$|E|^2$  [V$^2$/m$^2$]",
               r"$\sigma_a$  [S/m]"]
    subtitles = [r"$|H|^2(r)$", r"$|E|^2(r)$", r"$\sigma_a(r)$"]

    for ax, yl, st in zip(axes, ylabels, subtitles):
        ax.set_xlabel("r / R")
        ax.set_ylabel(yl)
        ax.set_title(st)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save_or_show(fig, save)
