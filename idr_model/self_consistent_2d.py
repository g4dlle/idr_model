"""
self_consistent_2d.py — самосогласованный 2D-решатель ВЧ-индукционного разряда.

Метод двух параметров (17032026.md), расширение на 2D:
  1. Для пробного n_e0 решаем уравнения Максвелла → профили E(r,z), H(r,z).
  2. Из профиля E вычисляем Da(r,z), νi(r,z) → собственное значение λ₀.
  3. Бисекцией по n_e0 находим n_e0*, при котором λ₀ = 1.

Физика λ₀ в 2D:
  (1/r)∂/∂r(r·Da·∂σ/∂r) + ∂/∂z(Da·∂σ/∂z) + (νi/λ₀²)·σ = 0

  bc_z="neumann" → осевые потери отсутствуют → λ₀ совпадает с 1D
  bc_z="dirichlet" → λ₀² ≈ Da·[(j₀₁/R)² + (π/L)²]/νi > 1D-значения

Зависимости: equations_2d, solver_2d, physics, config
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from equations_2d import make_grid_2d, build_sigma_equation_2d
from solver_2d import solve_idr_2d
from config import (
    N_GRID, N_Z, R_TUBE, L_TUBE, P_PA, H_WALL, BC_Z_SIGMA,
    MAX_ITER, TOL, RELAX,
)


# ---------------------------------------------------------------------------
# Вычисление собственного значения λ₀ для 2D уравнения баланса частиц
# ---------------------------------------------------------------------------

def compute_lambda0_2d(r: np.ndarray, z: np.ndarray,
                       hr: float, hz: float,
                       Da_2d: np.ndarray, nu_i_2d: np.ndarray,
                       bc_z_sigma: str = "dirichlet",
                       max_power_iter: int = 300,
                       tol: float = 1e-8) -> float:
    """
    Вычисляет λ₀² для 2D уравнения баланса частиц:

        (1/r)∂/∂r(r·Da·∂σ/∂r) + ∂/∂z(Da·∂σ/∂z) + (νi/λ₀²)·σ = 0

    Метод: степенная итерация.
        (-L)·σ_new = νi·σ_old,  λ₀² = 1/μ,
        μ = ∫∫σ·σ_new·r dr dz / ∫∫σ²·r dr dz

    Цилиндрический интеграл: ∫∫(·)·r dr dz через np.trapezoid.

    Parameters
    ----------
    r, z        : 1D-сетки (Nr+1,) и (Nz+1,)
    hr, hz      : шаги по r и z
    Da_2d       : (Nr+1, Nz+1)
    nu_i_2d     : (Nr+1, Nz+1)
    bc_z_sigma  : "dirichlet" или "neumann"

    Returns
    -------
    lambda0_sq : λ₀² > 0
    """
    Nr = len(r) - 1
    Nz = len(z) - 1
    annular = r[0] > 0.0

    # ── Начальное приближение, удовлетворяющее ГУ ────────────────────────
    if annular:
        r_profile = (r - r[0]) * (r[-1] - r)
        r_max = np.max(r_profile)
        r_profile = r_profile / r_max if r_max > 0 else r_profile
        r_profile[0]  = 0.0
    else:
        r_profile = np.maximum(1.0 - (r / r[-1])**2, 0.0)
    r_profile[-1] = 0.0

    if bc_z_sigma == "dirichlet":
        z_profile = z * (z[-1] - z)
        z_max = np.max(z_profile)
        z_profile = z_profile / z_max if z_max > 0 else z_profile
        z_profile[0]  = 0.0
        z_profile[-1] = 0.0
    else:
        z_profile = np.ones_like(z)

    sigma = np.outer(r_profile, z_profile)

    # ── Степенная итерация ────────────────────────────────────────────────
    r_col = r[:, np.newaxis]          # (Nr+1, 1) для broadcasting
    lambda0_sq_prev = 0.0

    for iteration in range(max_power_iter):
        A, rhs = build_sigma_equation_2d(
            r, z, hr, hz, Da_2d, nu_i_2d,
            sigma_ref=sigma, dt=None, bc_z_sigma=bc_z_sigma,
        )
        sigma_new = np.maximum(
            spsolve(A, rhs).reshape(Nr + 1, Nz + 1), 0.0
        )

        # Применяем ГУ явно (spsolve может дать ненулевые значения на BC-узлах)
        sigma_new[-1, :] = 0.0
        if annular:
            sigma_new[0, :] = 0.0
        if bc_z_sigma == "dirichlet":
            sigma_new[:, 0]  = 0.0
            sigma_new[:, -1] = 0.0

        # Коэффициент Рэлея: μ = ⟨σ, σ_new⟩ / ⟨σ, σ⟩  (цил. интеграл)
        norm_old = float(np.trapezoid(
            np.trapezoid(sigma**2 * r_col, z, axis=1), r))
        inner    = float(np.trapezoid(
            np.trapezoid(sigma * sigma_new * r_col, z, axis=1), r))

        if norm_old < 1e-300:
            return 1e30

        mu = inner / norm_old
        lambda0_sq = 1.0 / mu if abs(mu) > 1e-300 else 1e30

        # Нормируем sigma_new для следующей итерации
        norm_new = float(np.trapezoid(
            np.trapezoid(sigma_new**2 * r_col, z, axis=1), r))
        if norm_new > 1e-300:
            sigma = sigma_new * np.sqrt(norm_old / norm_new)
        else:
            sigma = sigma_new

        if (iteration > 0
                and abs(lambda0_sq - lambda0_sq_prev)
                / max(abs(lambda0_sq), 1e-30) < tol):
            break
        lambda0_sq_prev = lambda0_sq

    return lambda0_sq


# ---------------------------------------------------------------------------
# Решение уравнений Максвелла для заданного n_e0 (2D)
# ---------------------------------------------------------------------------

def solve_maxwell_for_ne0_2d(n_e0: float,
                              Nr: int = N_GRID,
                              Nz: int = N_Z,
                              R: float = R_TUBE,
                              L: float = L_TUBE,
                              p_pa: float = P_PA,
                              H_wall: float = H_WALL,
                              max_iter: int = MAX_ITER,
                              tol: float = TOL,
                              relax: float = RELAX,
                              r_inc: float = 0.0,
                              bc_z_sigma: str = BC_Z_SIGMA,
                              ) -> dict:
    """
    Обёртка вокруг solve_idr_2d: добавляет вычисление λ₀ в результат.

    Returns dict со всеми ключами solve_idr_2d плюс: lambda0
    """
    result = solve_idr_2d(
        Nr=Nr, Nz=Nz, R=R, L=L, p_pa=p_pa, H_wall=H_wall, n_e0=n_e0,
        max_iter=max_iter, tol=tol, relax=relax,
        r_inc=r_inc, bc_z_sigma=bc_z_sigma,
    )

    r, z, hr, hz = make_grid_2d(Nr, Nz, R, L, r_min=r_inc)

    lambda0_sq = compute_lambda0_2d(
        r, z, hr, hz,
        result["Da"], result["nu_i"],
        bc_z_sigma=bc_z_sigma,
    )
    result["lambda0"] = float(np.sqrt(max(lambda0_sq, 0.0)))
    return result


# ---------------------------------------------------------------------------
# Бисекция по n_e0 (2D)
# ---------------------------------------------------------------------------

def find_n_e0_2d(Nr: int = N_GRID,
                 Nz: int = N_Z,
                 R: float = R_TUBE,
                 L: float = L_TUBE,
                 p_pa: float = P_PA,
                 H_wall: float = H_WALL,
                 r_inc: float = 0.0,
                 bc_z_sigma: str = BC_Z_SIGMA,
                 n_e0_bounds: tuple = (1e14, 1e22),
                 tol_lambda: float = 0.01,
                 max_bisect: int = 50,
                 solver_kw: dict | None = None,
                 verbose: bool = False,
                 ) -> dict:
    """
    Бисекция в log-пространстве по n_e0: находит n_e0* с λ₀(n_e0) = 1.

    Логика полностью аналогична find_n_e0 из self_consistent.py,
    включая все исправления (bracket check, best_result из граничных
    точек, _adaptive_kw). Отличия только в параметрах сетки и
    вызове solve_maxwell_for_ne0_2d вместо solve_maxwell_for_ne0.

    Returns dict: n_e0, lambda0, converged, n_bisect, solution,
                  history, bracket_ok
    """
    kw_base = dict(max_iter=MAX_ITER, tol=TOL, relax=RELAX)
    if solver_kw is not None:
        kw_base.update(solver_kw)

    # Параметры сетки, фиксированные на всё время бисекции
    grid_kw = dict(Nr=Nr, Nz=Nz, R=R, L=L,
                   p_pa=p_pa, H_wall=H_wall,
                   r_inc=r_inc, bc_z_sigma=bc_z_sigma)

    log_lo = np.log10(n_e0_bounds[0])
    log_hi = np.log10(n_e0_bounds[1])
    history = []

    def _lam_effective(res: dict) -> float:
        return res["lambda0"] if res["converged"] else 1e10

    def _adaptive_kw(ne0: float) -> dict:
        kw = kw_base.copy()
        if ne0 > 1e19:
            kw['relax']    = min(kw.get('relax',    0.5),  0.2)
            kw['max_iter'] = max(kw.get('max_iter', 500), 1000)
        if ne0 > 1e20:
            kw['relax']    = min(kw.get('relax',    0.2),  0.1)
            kw['max_iter'] = max(kw.get('max_iter', 1000), 2000)
        return kw

    # ── Вычисление граничных точек ────────────────────────────────────────
    r_lo = solve_maxwell_for_ne0_2d(
        n_e0=10.0**log_lo, **grid_kw, **_adaptive_kw(10.0**log_lo))
    r_hi = solve_maxwell_for_ne0_2d(
        n_e0=10.0**log_hi, **grid_kw, **_adaptive_kw(10.0**log_hi))
    lam_lo = _lam_effective(r_lo)
    lam_hi = _lam_effective(r_hi)

    history.append((10.0**log_lo, r_lo["lambda0"]))
    history.append((10.0**log_hi, r_hi["lambda0"]))

    if verbose:
        print(f"  bracket: n_e0={10.0**log_lo:.3e} lam0={lam_lo:.4f}"
              f"  |  n_e0={10.0**log_hi:.3e} lam0={lam_hi:.4f}")

    # ── Инициализация best_result из граничных точек ──────────────────────
    best_result = None
    best_n_e0   = None
    best_lam0   = None

    for ne0_end, res_end in ((10.0**log_lo, r_lo), (10.0**log_hi, r_hi)):
        if res_end["converged"]:
            if best_result is None or abs(res_end["lambda0"] - 1.0) < abs(best_lam0 - 1.0):
                best_result = res_end
                best_n_e0   = ne0_end
                best_lam0   = res_end["lambda0"]

    # ── Граничная точка уже является корнем ──────────────────────────────
    if best_result is not None and abs(best_lam0 - 1.0) < tol_lambda:
        if verbose:
            print(f"  endpoint root: n_e0={best_n_e0:.3e} lam0={best_lam0:.6f}")
        return {
            "n_e0":       best_n_e0,
            "lambda0":    best_lam0,
            "converged":  True,
            "n_bisect":   len(history),
            "solution":   best_result,
            "history":    history,
            "bracket_ok": True,
        }

    # ── Проверка скобки ───────────────────────────────────────────────────
    bracket_ok = (lam_lo - 1.0) * (lam_hi - 1.0) <= 0.0
    if not bracket_ok:
        if best_result is None:
            if abs(lam_lo - 1.0) <= abs(lam_hi - 1.0):
                best_n_e0, best_lam0 = 10.0**log_lo, r_lo["lambda0"]
            else:
                best_n_e0, best_lam0 = 10.0**log_hi, r_hi["lambda0"]
        if verbose:
            print("  WARNING: bracket does not straddle lam0=1")
        return {
            "n_e0":       best_n_e0,
            "lambda0":    best_lam0,
            "converged":  False,
            "n_bisect":   len(history),
            "solution":   best_result,
            "history":    history,
            "bracket_ok": False,
        }

    # ── Бисекция ──────────────────────────────────────────────────────────
    n_e0_trial = 10.0**(0.5 * (log_lo + log_hi))
    lam0       = 0.5 * (lam_lo + lam_hi)

    for step in range(max_bisect):
        log_mid    = 0.5 * (log_lo + log_hi)
        n_e0_trial = 10.0**log_mid

        result = solve_maxwell_for_ne0_2d(
            n_e0=n_e0_trial, **grid_kw, **_adaptive_kw(n_e0_trial))
        lam0           = result["lambda0"]
        lam0_effective = _lam_effective(result)

        history.append((n_e0_trial, lam0))

        if verbose:
            print(f"  bisect {step:3d}: n_e0={n_e0_trial:.3e}  lam0={lam0:.6f}"
                  f"  conv={result['converged']}  iters={result['n_iter']}")

        if result["converged"]:
            if best_result is None or abs(lam0 - 1.0) < abs(best_lam0 - 1.0):
                best_result = result
                best_n_e0   = n_e0_trial
                best_lam0   = lam0

        if result["converged"] and abs(lam0 - 1.0) < tol_lambda:
            break

        if (lam_lo - 1.0) * (lam0_effective - 1.0) > 0.0:
            log_lo = log_mid
            lam_lo = lam0_effective
        else:
            log_hi = log_mid
            lam_hi = lam0_effective

    if best_result is None:
        best_n_e0 = n_e0_trial
        best_lam0 = lam0

    return {
        "n_e0":       best_n_e0,
        "lambda0":    best_lam0,
        "converged":  best_result is not None and abs(best_lam0 - 1.0) < tol_lambda,
        "n_bisect":   len(history),
        "solution":   best_result,
        "history":    history,
        "bracket_ok": True,
    }


# ---------------------------------------------------------------------------
# Верхнеуровневая функция
# ---------------------------------------------------------------------------

def solve_self_consistent_2d(Nr: int = N_GRID,
                              Nz: int = N_Z,
                              R: float = R_TUBE,
                              L: float = L_TUBE,
                              p_pa: float = P_PA,
                              H_wall: float = H_WALL,
                              r_inc: float = 0.0,
                              bc_z_sigma: str = BC_Z_SIGMA,
                              n_e0_bounds: tuple = (1e14, 1e22),
                              tol_lambda: float = 0.01,
                              max_bisect: int = 50,
                              verbose: bool = False,
                              **solver_kw) -> dict:
    """
    Верхнеуровневая функция. Находит E_R и n_e0* для 2D-разряда.

    E_R вычисляется как |E_φ| на стенке r=R в середине трубки по z.
    """
    result = find_n_e0_2d(
        Nr=Nr, Nz=Nz, R=R, L=L, p_pa=p_pa, H_wall=H_wall,
        r_inc=r_inc, bc_z_sigma=bc_z_sigma,
        n_e0_bounds=n_e0_bounds, tol_lambda=tol_lambda,
        max_bisect=max_bisect,
        solver_kw=solver_kw if solver_kw else None,
        verbose=verbose,
    )
    if result["solution"] is not None:
        sol = result["solution"]
        # |E_φ|(R, z_mid): стенка r=R, середина трубки по z
        z_mid = sol["v"].shape[1] // 2
        result["E_R"] = float(np.sqrt(max(sol["v"][-1, z_mid], 0.0)))
    else:
        result["E_R"] = 0.0
    return result
