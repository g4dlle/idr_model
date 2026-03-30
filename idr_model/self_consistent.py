"""
self_consistent.py — самосогласованный решатель ВЧ-индукционного разряда.

Метод двух параметров (17032026.md):
  1. Для пробного n_e0 решаем уравнения Максвелла → профили E(r), H(r).
  2. Из профиля E(r) вычисляем Da(r), νi(r) → собственное значение λ₀.
  3. Бисекцией по n_e0 находим n_e0*, при котором λ₀ = 1.

Зависимости: solver.py, equations.py, physics.py, config.py
"""

import numpy as np
from equations import make_grid, build_H_equation, build_sigma_equation
from solver import thomas_solve, compute_alpha
from physics import (
    conductivity, effective_field,
    ambipolar_diffusion, ionization_freq,
)
from boundary import apply_wall_sigma
from config import (
    N_GRID, R_TUBE, P_PA, H_WALL, N_E0,
    MAX_ITER, TOL, RELAX,
    OMEGA,
)


MU_0 = 4.0 * np.pi * 1e-7


# ---------------------------------------------------------------------------
# Вычисление собственного значения λ₀
# ---------------------------------------------------------------------------

def compute_lambda0(r: np.ndarray, h: float,
                    Da: np.ndarray, nu_i: np.ndarray,
                    max_power_iter: int = 300,
                    tol: float = 1e-8) -> float:
    """
    Вычисляет собственное значение λ₀² для уравнения баланса частиц:

        (1/r) d/dr(r · Da · dσ/dr) + (νi / λ₀²) · σ = 0
        σ(R) = 0,  dσ/dr(0) = 0

    Метод: степенная итерация (power iteration).
    λ₀² = 1/μ, где μ — доминирующее собственное значение M = (-Da·Δ)⁻¹·diag(νi).

    Returns
    -------
    lambda0_sq : λ₀² (> 0)
    """
    N = len(r) - 1
    sigma = np.maximum(1.0 - (r / r[-1])**2, 0.0)
    sigma[-1] = 0.0

    lambda0_sq_prev = 0.0

    for iteration in range(max_power_iter):
        l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma)
        sigma_new = thomas_solve(l, m, up, rhs)
        sigma_new = np.maximum(sigma_new, 0.0)
        sigma_new[-1] = 0.0

        norm_old = np.trapezoid(sigma[:-1]**2 * r[:-1], r[:-1])
        inner_product = np.trapezoid(sigma[:-1] * sigma_new[:-1] * r[:-1], r[:-1])

        if norm_old < 1e-300:
            return 1e30

        mu = inner_product / norm_old
        lambda0_sq = 1.0 / mu if abs(mu) > 1e-300 else 1e30

        norm_new = np.trapezoid(sigma_new[:-1]**2 * r[:-1], r[:-1])
        if norm_new > 1e-300:
            sigma = sigma_new * np.sqrt(norm_old / norm_new)
        else:
            sigma = sigma_new

        if iteration > 0 and abs(lambda0_sq - lambda0_sq_prev) / max(abs(lambda0_sq), 1e-30) < tol:
            break
        lambda0_sq_prev = lambda0_sq

    return lambda0_sq


# ---------------------------------------------------------------------------
# Решение уравнений Максвелла для заданного n_e0
# ---------------------------------------------------------------------------

def solve_maxwell_for_ne0(n_e0: float,
                          N: int = N_GRID,
                          R: float = R_TUBE,
                          p_pa: float = P_PA,
                          H_wall: float = H_WALL,
                          max_iter: int = MAX_ITER,
                          tol: float = TOL,
                          relax: float = RELAX,
                          ) -> dict:
    """
    Решает уравнения Максвелла самосогласованно с профилем n_e(r),
    при фиксированной АБСОЛЮТНОЙ амплитуде n_e0 = n_e(0).

    Ключевое отличие от solve_idr: амплитуда n_e0 НЕ нормируется.
    Профиль n_e(r) = n_e0·n̄(r) итерируется по форме n̄(r),
    а σ(r) вычисляется из n_e(r) по формуле проводимости.

    Returns dict с ключами: r, u, v, sigma_a, sigma_p, n_e, Da, nu_i,
                            lambda0, converged, n_iter, residuals
    """
    r, h = make_grid(N, R)

    # Начальный профиль n_e: J₀-подобный, амплитуда = n_e0
    n_e_shape = np.maximum(1.0 - (r / R)**2, 0.0)
    n_e_shape[-1] = 0.0
    n_e = n_e0 * n_e_shape

    sigma_a, sigma_p, _ = conductivity(n_e, p_pa)

    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    u[-1] = H_wall**2

    residuals = []
    converged = False

    for iteration in range(max_iter):
        # (a) Solve for |H|²
        alpha = compute_alpha(sigma_a, sigma_p)
        if np.all(alpha == 0):
            alpha = np.full(N + 1, 1e-10)

        l, m, up, rhs_h = build_H_equation(r, h, alpha, sigma_a, v, H_wall**2)
        u_new = thomas_solve(l, m, up, rhs_h)
        u_new = np.maximum(u_new, 0.0)

        # (b) E_phi из закона Фарадея
        H_abs = np.sqrt(np.maximum(u_new, 0.0))
        integrand = H_abs * r
        intervals = 0.5 * h * (integrand[:-1] + integrand[1:])
        integrals = np.zeros(N + 1)
        integrals[1:] = np.cumsum(intervals)
        v_new = np.zeros(N + 1)
        v_new[1:] = (OMEGA * MU_0 * integrals[1:] / r[1:])**2

        # (c) E_eff, Da, νi
        E_abs = np.sqrt(v_new)
        E_eff = effective_field(E_abs, p_pa)
        Da   = ambipolar_diffusion(E_eff, p_pa)
        nu_i = ionization_freq(E_eff, p_pa)

        # (d) Обновление ФОРМЫ профиля n_e через power iteration на σ.
        #     Используем Da, νi из текущего поля.
        #     Power iteration: (-Da·Δ)·σ_new = νi·σ_old
        #     σ_old  = σ_a (текущий).
        l_s, m_s, up_s, rhs_s = build_sigma_equation(r, h, Da, nu_i,
                                                       sigma_a_ref=sigma_a)
        sigma_raw = thomas_solve(l_s, m_s, up_s, rhs_s)
        sigma_raw = np.maximum(sigma_raw, 0.0)
        apply_wall_sigma(sigma_raw)

        # Извлекаем ФОРМУ профиля: нормируем к макс. = 1
        shape_max = np.max(sigma_raw[:-1])
        if shape_max > 1e-300:
            n_e_shape_new = sigma_raw / shape_max
        else:
            n_e_shape_new = np.maximum(1.0 - (r / R)**2, 0.0)
        n_e_shape_new[-1] = 0.0

        # n_e с фиксированной амплитудой n_e0
        n_e_new = n_e0 * n_e_shape_new

        # σ из n_e
        sigma_a_new, sigma_p_new, _ = conductivity(n_e_new, p_pa)

        # (e) Под-релаксация
        sigma_a_next = relax * sigma_a_new + (1.0 - relax) * sigma_a
        sigma_p_next = relax * sigma_p_new + (1.0 - relax) * sigma_p

        # (f) Сходимость: отслеживаем изменение формы + полей
        s_max_new = max(np.max(sigma_a_new[:-1]), 1e-300)
        s_max_old = max(np.max(sigma_a[:-1]), 1e-300)
        sh_new = sigma_a_new[:-1] / s_max_new
        sh_old = sigma_a[:-1] / s_max_old
        res_shape = float(np.max(np.abs(sh_new - sh_old)))

        if np.any(u[:-1] > 0):
            safe_u = np.where(np.abs(u[:-1]) > 0, np.abs(u[:-1]), 1e-300)
            res_u = float(np.max(np.abs(u_new[:-1] - u[:-1]) / safe_u))
        else:
            res_u = 1.0

        res = max(res_u, res_shape)
        residuals.append(res)

        u[:] = u_new
        v[:] = v_new
        sigma_a[:] = sigma_a_next
        sigma_p[:] = sigma_p_next
        n_e_shape = n_e_shape_new

        if res < tol:
            converged = True
            break

    # Финальные величины
    E_eff_fin  = effective_field(np.sqrt(v), p_pa)
    Da_final   = ambipolar_diffusion(E_eff_fin, p_pa)
    nu_i_final = ionization_freq(E_eff_fin, p_pa)
    n_e_final  = n_e0 * n_e_shape
    n_e_final[-1] = 0.0

    # λ₀
    lambda0_sq = compute_lambda0(r, h, Da_final, nu_i_final)
    lambda0 = np.sqrt(max(lambda0_sq, 0.0))

    return {
        "r":         r,
        "u":         u,
        "v":         v,
        "sigma_a":   sigma_a,
        "sigma_p":   sigma_p,
        "n_e":       n_e_final,
        "Da":        Da_final,
        "nu_i":      nu_i_final,
        "lambda0":   lambda0,
        "converged": converged,
        "n_iter":    len(residuals),
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Бисекция по n_e0
# ---------------------------------------------------------------------------

def find_n_e0(N: int = N_GRID,
              R: float = R_TUBE,
              p_pa: float = P_PA,
              H_wall: float = H_WALL,
              n_e0_bounds: tuple[float, float] = (1e14, 1e22),
              tol_lambda: float = 0.01,
              max_bisect: int = 50,
              solver_kw: dict | None = None,
              verbose: bool = False,
              ) -> dict:
    """
    Бисекция в log-пространстве по n_e0: находит n_e0* с λ₀(n_e0) = 1.

    Физика: 
    λ₀² стоит в знаменателе: div(Da grad n) + (νi / λ₀²) n = 0.
    - При λ₀ < 1: мы искусственно "увеличили" νi (поделив на λ₀²<1),
      значит реальная ионизация слишком слаба (ионизация < диффузии).
      Нужно увеличить n_e0 → усилить скин-эффект → E вытесняется к стенке,
      но из-за резкого градиента H там E становится огромным → скачок ионизации.
    - При λ₀ > 1: реальная ионизация избыточна (ионизация > диффузии).
      Нужно уменьшить n_e0.

    Returns dict: n_e0, lambda0, converged, n_bisect, solution, history
    """
    kw_base = dict(max_iter=MAX_ITER, tol=TOL, relax=RELAX)
    if solver_kw is not None:
        kw_base.update(solver_kw)

    log_lo = np.log10(n_e0_bounds[0])
    log_hi = np.log10(n_e0_bounds[1])
    history = []
    best_result = None
    n_e0_trial = 10.0**(0.5*(log_lo+log_hi))
    lam0 = 0.0

    for step in range(max_bisect):
        log_mid = 0.5 * (log_lo + log_hi)
        n_e0_trial = 10.0**log_mid

        # Адаптивная релаксация: чем выше n_e0 (сильнее скин), тем меньше relax
        kw = kw_base.copy()
        if n_e0_trial > 1e19:
            kw['relax'] = min(kw.get('relax', 0.5), 0.2)
            kw['max_iter'] = max(kw.get('max_iter', 500), 1000)
        if n_e0_trial > 1e20:
            kw['relax'] = min(kw.get('relax', 0.2), 0.1)
            kw['max_iter'] = max(kw.get('max_iter', 1000), 2000)

        result = solve_maxwell_for_ne0(
            n_e0=n_e0_trial, N=N, R=R, p_pa=p_pa, H_wall=H_wall, **kw)
        lam0 = result["lambda0"]

        # Если внутренний решатель не сошёлся, λ₀ ненадёжен.
        # Физически: нет сходимости → σ слишком велика → скин-эффект
        # коллапсировал → это режим λ₀ >> 1. Трактуем как λ₀ > 1.
        if not result["converged"]:
            lam0_effective = 1e10   # большое число → уменьшить n_e0
        else:
            lam0_effective = lam0
            best_result = result    # обновляем только при сходимости

        history.append((n_e0_trial, lam0))

        if verbose:
            print(f"  bisect {step:3d}: n_e0={n_e0_trial:.3e}  lambda0={lam0:.6f}"
                  f"  conv={result['converged']}  iters={result['n_iter']}")

        if result["converged"] and abs(lam0 - 1.0) < tol_lambda:
            break

        # λ₀ < 1 → ионизация > диффузии → увеличить n_e0 (усилить скин)
        # λ₀ > 1 или нет сходимости → уменьшить n_e0
        if lam0_effective < 1.0:
            log_lo = log_mid
        else:
            log_hi = log_mid

    return {
        "n_e0":      n_e0_trial,
        "lambda0":   lam0,
        "converged": abs(lam0 - 1.0) < tol_lambda,
        "n_bisect":  len(history),
        "solution":  best_result,
        "history":   history,
    }


def solve_self_consistent(N: int = N_GRID, R: float = R_TUBE,
                          p_pa: float = P_PA, H_wall: float = H_WALL,
                          n_e0_bounds: tuple[float, float] = (1e14, 1e22),
                          tol_lambda: float = 0.01, max_bisect: int = 50,
                          verbose: bool = False, **solver_kw) -> dict:
    """Верхнеуровневая функция. Находит E_R и n_e0."""
    result = find_n_e0(N=N, R=R, p_pa=p_pa, H_wall=H_wall,
                       n_e0_bounds=n_e0_bounds, tol_lambda=tol_lambda,
                       max_bisect=max_bisect,
                       solver_kw=solver_kw if solver_kw else None,
                       verbose=verbose)
    if result["solution"] is not None:
        result["E_R"] = np.sqrt(max(result["solution"]["v"][-1], 0.0))
    else:
        result["E_R"] = 0.0
    return result
