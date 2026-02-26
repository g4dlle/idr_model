"""
postprocess.py -- post-processing and visualization for the 1D IDR model.

Public API
----------
joule_dissipation(r, sigma_a, sigma_mod2, u, v)  -> Q array
total_power(r, Q)                                 -> float [W/m]
summary(result, p_pa)                             -> print

plot_dashboard(result, p_pa, save)         -- 2x3 panel: all fields + convergence
plot_sigma_norm(results, labels, save)     -- sigma/sigma0 vs r/R (Fig. 1 style)
plot_fields_comparison(results, labels)    -- H, E, sigma on 3 axes
plot_joule(result, save)                   -- Q(r) dissipation profile
plot_convergence(result, save)             -- residual vs iteration (semilog)
plot_parametric(param_values, results...)  -- parametric sweep curves
"""

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Style settings
_STYLE = {
    "figure.dpi":        120,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "lines.linewidth":   2.0,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


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
# Physical computations
# ---------------------------------------------------------------------------

def joule_dissipation(r, sigma_a, sigma_mod2, u, v):
    """
    Specific Joule dissipation, time-averaged [W/m^3].

    Q(r) = (1/2) * sigma_a * |E_amplitude|^2 = 0.5 * sigma_a * v

    Физика: для синусоидального поля E(t) = |E_amp|·cos(ωt):
        <q(t)> = sigma_a · <E²(t)> = sigma_a · |E_amp|²/2 = (1/2)·sigma_a·v
    Множитель 1/2 обязателен, так как v = |E_amplitude|² (квадрат амплитуды,
    а не среднеквадратичного значения).

    Параметры sigma_mod2 и u оставлены для совместимости сигнатуры, не используются.

    Parameters
    ----------
    r          : grid (N+1,)
    sigma_a    : active conductivity (N+1,) [S/m]
    sigma_mod2 : |sigma|^2 (N+1,)  [не используется]
    u          : |H|^2 (N+1,) [A^2/m^2]  [не используется]
    v          : |E_amplitude|^2 (N+1,) [V^2/m^2]

    Returns
    -------
    Q : time-averaged specific power dissipation (N+1,) [W/m^3]
    """
    return 0.5 * sigma_a * v


def total_power(r, Q):
    """
    Power per unit tube length [W/m]:
    P = 2*pi * integral_0^R Q(r)*r dr   (trapezoidal rule)
    """
    return 2.0 * np.pi * np.trapezoid(Q * r, r)


def summary(result, p_pa):
    """Print key solution characteristics."""
    r       = result["r"]
    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]
    u       = result["u"]
    v       = result["v"]

    mod2 = sigma_a**2 + sigma_p**2
    Q    = joule_dissipation(r, sigma_a, mod2, u, v)
    P    = total_power(r, Q)

    print("=" * 52)
    print("  1D IDR model results")
    print("=" * 52)
    print(f"  Pressure:          p = {p_pa:.1f} Pa")
    print(f"  Tube radius:       R = {r[-1]*1e3:.1f} mm")
    print(f"  Converged:         {'yes' if result['converged'] else 'no'}")
    print(f"  Iterations:        {result['n_iter']}")
    print(f"  sigma_a(0):        {sigma_a[0]:.4e} S/m")
    print(f"  sigma_a(R):        {sigma_a[-1]:.4e} S/m")
    print(f"  |H|^2(R):          {u[-1]:.4e} A^2/m^2")
    print(f"  |E|^2_max:         {v.max():.4e} V^2/m^2")
    print(f"  Power per length:  {P:.4e} W/m")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Dashboard: all fields + convergence in one figure
# ---------------------------------------------------------------------------

def plot_dashboard(result, p_pa=None, title=None, save=None):
    """
    2x3 panel: |H|^2, |E|^2, sigma_a, n_e, Q(r), convergence history.

    Parameters
    ----------
    result : dict from solve_idr
    p_pa   : pressure [Pa] (used in the figure title)
    title  : custom figure title (None => auto)
    save   : file path to save (PNG/PDF), None => show interactively
    """
    if not HAS_MPL:
        print("[postprocess] matplotlib not installed.")
        return
    _apply_style()

    r       = result["r"]
    R       = r[-1]
    rn      = r / R
    u       = result["u"]
    v       = result["v"]
    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]
    n_e     = result["n_e"]
    mod2    = sigma_a**2 + sigma_p**2
    Q       = joule_dissipation(r, sigma_a, mod2, u, v)
    res_list = result["residuals"]

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # |H|^2
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(rn, u, color=_COLORS[0])
    ax0.set_xlabel("r / R")
    ax0.set_ylabel(r"$|H|^2$  [A$^2$/m$^2$]")
    ax0.set_title(r"Magnetic field  $|H|^2(r)$")
    ax0.set_xlim(0, 1)

    # |E|^2
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(rn, v, color=_COLORS[1])
    ax1.set_xlabel("r / R")
    ax1.set_ylabel(r"$|E|^2$  [V$^2$/m$^2$]")
    ax1.set_title(r"Electric field  $|E|^2(r)$")
    ax1.set_xlim(0, 1)

    # sigma_a
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(rn, sigma_a, color=_COLORS[2])
    ax2.set_xlabel("r / R")
    ax2.set_ylabel(r"$\sigma_a$  [S/m]")
    ax2.set_title(r"Active conductivity  $\sigma_a(r)$")
    ax2.set_xlim(0, 1)

    # n_e
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rn, n_e, color=_COLORS[3])
    ax3.set_xlabel("r / R")
    ax3.set_ylabel(r"$n_e$  [m$^{-3}$]")
    ax3.set_title(r"Electron density  $n_e(r)$")
    ax3.set_xlim(0, 1)
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Q(r)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(rn, Q, color=_COLORS[4])
    ax4.fill_between(rn, Q, alpha=0.15, color=_COLORS[4])
    ax4.set_xlabel("r / R")
    ax4.set_ylabel(r"$Q$  [W/m$^3$]")
    ax4.set_title(r"Joule dissipation  $Q(r)$")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(bottom=0)
    P_val = total_power(r, Q)
    ax4.text(0.97, 0.96, f"P = {P_val:.2e} W/m",
             transform=ax4.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.25",
                                   fc="white", alpha=0.85))

    # Convergence
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.semilogy(res_list, color=_COLORS[5])
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Residual")
    ax5.set_title("Convergence history")
    ok_str = "converged" if result["converged"] else "NOT converged"
    ax5.text(0.97, 0.97, ok_str,
             transform=ax5.transAxes, ha="right", va="top",
             fontsize=9, color="#2ca02c" if result["converged"] else "#d62728")

    if title is None and p_pa is not None:
        title = (f"1D IDR model (Ar)  |  "
                 f"p = {p_pa:.0f} Pa,  R = {R*1e3:.1f} mm")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Normalised conductivity sigma/sigma_0  (Fig. 1 style)
# ---------------------------------------------------------------------------

def plot_sigma_norm(results, labels=None,
                    title=r"Normalised conductivity  $\sigma / \sigma_0$",
                    sigma_0_list=None, add_bessel=True, save=None):
    """
    Plot sigma(r)/sigma_0 for one or several solutions.

    Parameters
    ----------
    results      : one dict or a list of dicts from solve_idr
    labels       : curve labels
    sigma_0_list : reference sigma_0 per curve (None => use sigma_a[0])
    add_bessel   : overlay J_0(2.4048 * r/R) as the analytical approximation
    save         : file path to save, None => show
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

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (res, lbl, s0) in enumerate(zip(results, labels, sigma_0_list)):
        r  = res["r"]
        rn = r / r[-1]
        sa = res["sigma_a"]
        if s0 is None or s0 == 0.0:
            s0 = sa[0] if sa[0] > 0 else 1.0
        ax.plot(rn, sa / s0,
                color=_COLORS[i % len(_COLORS)],
                label=lbl, zorder=3)

    if add_bessel:
        try:
            from scipy.special import j0
            rn_fine  = np.linspace(0, 1, 300)
            j0_vals  = np.maximum(j0(2.4048 * rn_fine), 0.0)
            ax.plot(rn_fine, j0_vals, "k--", lw=1.4, alpha=0.65,
                    label=r"$J_0(2.405\,r/R)$  [analytic]")
        except ImportError:
            pass

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("r / R", fontsize=12)
    ax.set_ylabel(r"$\sigma_a / \sigma_0$", fontsize=12)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Side-by-side field comparison
# ---------------------------------------------------------------------------

def plot_fields_comparison(results, labels=None,
                            title="Field profiles comparison",
                            save=None):
    """
    Three subplots: |H|^2, |E|^2, sigma_a for several solutions.

    Useful for comparing results at different pressures or H_wall values.
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
        rn = res["r"] / res["r"][-1]
        c  = _COLORS[i % len(_COLORS)]
        axes[0].plot(rn, res["u"],       color=c, label=lbl)
        axes[1].plot(rn, res["v"],       color=c, label=lbl)
        axes[2].plot(rn, res["sigma_a"], color=c, label=lbl)

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


# ---------------------------------------------------------------------------
# Joule dissipation profile
# ---------------------------------------------------------------------------

def plot_joule(result, p_pa=None, save=None):
    """
    Plot Q(r) with shaded area and total power annotation.

    Parameters
    ----------
    result : dict from solve_idr
    p_pa   : pressure [Pa] (used in title)
    save   : file path, None => show
    """
    if not HAS_MPL:
        return
    _apply_style()

    r       = result["r"]
    rn      = r / r[-1]
    sigma_a = result["sigma_a"]
    sigma_p = result["sigma_p"]
    mod2    = sigma_a**2 + sigma_p**2
    Q       = joule_dissipation(r, sigma_a, mod2, result["u"], result["v"])
    P       = total_power(r, Q)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rn, Q, color=_COLORS[4], zorder=3)
    ax.fill_between(rn, Q, alpha=0.18, color=_COLORS[4])
    ax.set_xlabel("r / R")
    ax.set_ylabel(r"$Q$  [W/m$^3$]")
    ttl = r"Joule dissipation  $Q(r)$"
    if p_pa is not None:
        ttl += f"   (p = {p_pa:.0f} Pa)"
    ax.set_title(ttl)
    ax.text(0.97, 0.97, f"P = {P:.3e} W/m",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Convergence history
# ---------------------------------------------------------------------------

def plot_convergence(results, labels=None, tol=None, save=None):
    """
    Semilog plot of residual vs iteration for one or several runs.

    Parameters
    ----------
    results : dict or list of dicts from solve_idr
    labels  : curve labels
    tol     : tolerance threshold (drawn as dashed line)
    save    : file path, None => show
    """
    if not HAS_MPL:
        return
    _apply_style()

    if isinstance(results, dict):
        results = [results]
    if labels is None:
        labels = [f"run {i+1}" for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (res, lbl) in enumerate(zip(results, labels)):
        ax.semilogy(res["residuals"],
                    color=_COLORS[i % len(_COLORS)],
                    label=lbl, zorder=3)

    if tol is not None:
        ax.axhline(tol, color="black", lw=1.2, ls="--",
                   label=f"tol = {tol:.0e}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Residual  $\max|\Delta u / u|$")
    ax.set_title("Convergence of the iterative solver")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Parametric sweep: family of curves
# ---------------------------------------------------------------------------

def plot_parametric(param_values, results,
                    param_name="p", param_unit="Pa",
                    field="sigma_norm",
                    title=None, save=None):
    """
    Plot a family of field profiles for different parameter values.

    Parameters
    ----------
    param_values : list of numeric parameter values
    results      : list of dicts from solve_idr, one per parameter value
    param_name   : parameter name for legend ("p", "R", "H_wall", ...)
    param_unit   : unit string ("Pa", "mm", "A/m", ...)
    field        : one of "sigma_norm" | "u" | "v" | "sigma_a" | "n_e"
    title        : figure title (None => auto)
    save         : file path, None => show
    """
    if not HAS_MPL:
        return
    _apply_style()

    _field_labels = {
        "sigma_norm": r"$\sigma_a / \sigma_a(0)$",
        "u":          r"$|H|^2 / |H|^2_R$",
        "v":          r"$|E|^2 / |E|^2_{\max}$",
        "sigma_a":    r"$\sigma_a$  [S/m]",
        "n_e":        r"$n_e$  [m$^{-3}$]",
    }
    if field not in _field_labels:
        raise ValueError(f"Unknown field '{field}'. "
                         f"Choose from: {list(_field_labels)}")

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (pv, res) in enumerate(zip(param_values, results)):
        rn  = res["r"] / res["r"][-1]
        c   = _COLORS[i % len(_COLORS)]
        lbl = f"{param_name} = {pv} {param_unit}"

        if field == "sigma_norm":
            sa = res["sigma_a"]
            s0 = sa[0] if sa[0] > 0 else 1.0
            y  = sa / s0
        elif field == "u":
            H2R = res["u"][-1] if res["u"][-1] > 0 else 1.0
            y   = res["u"] / H2R
        elif field == "v":
            vmax = res["v"].max() if res["v"].max() > 0 else 1.0
            y    = res["v"] / vmax
        elif field == "sigma_a":
            y = res["sigma_a"]
        else:   # n_e
            y = res["n_e"]

        ax.plot(rn, y, color=c, label=lbl)

    ax.set_xlabel("r / R", fontsize=12)
    ax.set_ylabel(_field_labels[field], fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    if title is None:
        title = f"{_field_labels[field]}  vs  {param_name}"
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save)
