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
    bohm_velocity,
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
                    tol: float = 1e-8,
                    v_bohm: float = 0.0,
                    i_wall: int | None = None) -> float:
    """
    Вычисляет собственное значение λ₀² для уравнения баланса частиц:

        (1/r) d/dr(r · Da · dσ/dr) + (νi / λ₀²) · σ = 0

    Граничные условия:
        dσ/dr(0) = 0  (симметрия)
        σ(r[i_wall]) = 0   (Дирихле, по умолчанию i_wall = N → r=R)
        σ[i] = 0 для i > i_wall  (оболочка)

    Метод: степенная итерация (power iteration).
    λ₀² = μ, где μ — доминирующее собственное значение M = (-Da·Δ)⁻¹·diag(νi).

    Returns
    -------
    lambda0_sq : λ₀² (> 0)
    """
    N = len(r) - 1
    if i_wall is None:
        i_wall = N
    i_wall = max(1, min(i_wall, N))

    sigma = np.maximum(1.0 - (r / r[i_wall])**2, 0.0)
    sigma[i_wall:] = 0.0   # оболочка + стенка

    lambda0_sq_prev = 0.0

    for iteration in range(max_power_iter):
        l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i,
                                              sigma_a_ref=sigma,
                                              v_bohm=v_bohm,
                                              i_wall=i_wall)
        sigma_new = thomas_solve(l, m, up, rhs)
        sigma_new = np.maximum(sigma_new, 0.0)
        sigma_new[i_wall:] = 0.0   # принудительно обнуляем оболочку

        # Интегрируем только по плазменной области [0, r[i_wall])
        norm_old      = np.trapezoid(sigma[:i_wall]**2 * r[:i_wall], r[:i_wall])
        inner_product = np.trapezoid(sigma[:i_wall] * sigma_new[:i_wall] * r[:i_wall],
                                     r[:i_wall])

        if norm_old < 1e-300:
            return 1e30

        mu = inner_product / norm_old
        # mu — доминирующее собственное значение A⁻¹·B, где A=−Da·Δ, B=diag(νi).
        # λ₀² = γ = mu. При пороге (νi = Da·λ₁²/R²): mu = 1 → λ₀ = 1. ✓
        lambda0_sq = mu if abs(mu) > 1e-300 else 1e-30

        norm_new = np.trapezoid(sigma_new[:i_wall]**2 * r[:i_wall], r[:i_wall])
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
                          v_bohm: float = 0.0,
                          delta_sheath: float = 0.0,
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

    # Индекс граничного узла плазмы (оболочка: r > r[i_wall])
    # delta_sheath = 0 → i_wall = N (стандартное ГУ на стенке)
    # delta_sheath > 0 → плазма ограничена r_eff = R − delta_sheath
    if delta_sheath > 0.0:
        r_eff  = R - delta_sheath
        i_wall = int(np.searchsorted(r, r_eff, side='right')) - 1
        i_wall = max(1, min(i_wall, N - 1))
    else:
        i_wall = N

    # Начальный профиль n_e: параболический, амплитуда = n_e0
    n_e_shape = np.maximum(1.0 - (r / r[i_wall])**2, 0.0)
    n_e_shape[i_wall:] = 0.0
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

        # В оболочке (i >= i_wall) плазмы нет → H = H_wall (вакуум).
        # Чтобы матрица H-уравнения была невырожденной, используем
        # Дирихле на i_wall и тождественные строки для i > i_wall.
        # Для этого патчим alpha: в шeathe ставим малое значение;
        # после решения перекрываем u в шeathe значением H_wall².
        if i_wall < N:
            alpha[i_wall:] = np.maximum(alpha[i_wall:], 1e-12)

        l, m, up, rhs_h = build_H_equation(r, h, alpha, sigma_a, v, H_wall**2)

        # Дирихле H² = H_wall² на i_wall (граница плазма–оболочка)
        if i_wall < N:
            l[i_wall] = 0.0
            m[i_wall] = 1.0
            up[i_wall] = 0.0
            rhs_h[i_wall] = H_wall**2
            # Для i > i_wall: u[i] = H_wall² (вакуум, H не изменяется)
            for i in range(i_wall + 1, N + 1):
                l[i] = 0.0
                m[i] = 1.0
                up[i] = 0.0
                rhs_h[i] = H_wall**2

        u_new = thomas_solve(l, m, up, rhs_h)
        u_new = np.maximum(u_new, 0.0)
        # Явно фиксируем H² в оболочке
        if i_wall < N:
            u_new[i_wall:] = H_wall**2

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
        #     Power iteration: (-Da·Δ)·σ_new = νi·σ_old
        #     При v_bohm > 0: ГУ 3-го рода на стенке (σ_new[-1] ≠ 0).
        l_s, m_s, up_s, rhs_s = build_sigma_equation(r, h, Da, nu_i,
                                                       sigma_a_ref=sigma_a,
                                                       v_bohm=v_bohm,
                                                       i_wall=i_wall)
        sigma_raw = thomas_solve(l_s, m_s, up_s, rhs_s)
        sigma_raw = np.maximum(sigma_raw, 0.0)
        sigma_raw[i_wall:] = 0.0   # оболочка всегда обнулена
        if v_bohm == 0.0 and i_wall == N:
            apply_wall_sigma(sigma_raw)   # Дирихле на физической стенке

        # Извлекаем ФОРМУ профиля: нормируем к максимуму внутри плазмы
        shape_max = np.max(sigma_raw[:i_wall]) if i_wall > 0 else 0.0
        if shape_max > 1e-300:
            n_e_shape_new = sigma_raw / shape_max
        else:
            n_e_shape_new = np.maximum(1.0 - (r / r[i_wall])**2, 0.0)
        n_e_shape_new[i_wall:] = 0.0   # оболочка

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

        # Используем общую шкалу max(|u|, |u_new|) чтобы избежать деления
        # на малые/нулевые значения (overflow при u→0 в первых итерациях).
        u_ref = max(np.max(np.abs(u_new[:-1])), np.max(np.abs(u[:-1])), 1e-30)
        res_u = float(np.max(np.abs(u_new[:-1] - u[:-1])) / u_ref)

        res = max(res_u, res_shape)
        residuals.append(res)

        u[:] = u_new
        v[:] = v_new
        sigma_a[:] = sigma_a_next
        sigma_p[:] = sigma_p_next
        # n_e_shape не обновляем из сырого sigma_raw — он будет пересчитан
        # из финального (релаксированного) sigma_a после цикла.

        if res < tol:
            converged = True
            break

    # Финальные величины.
    # n_e восстанавливается из релаксированного sigma_a (σ ∝ n_e при фикс. E).
    sigma_a_fin_max = np.max(sigma_a[:i_wall]) if i_wall > 0 else 0.0
    if sigma_a_fin_max > 1e-300:
        n_e_final = n_e0 * sigma_a / sigma_a_fin_max
    else:
        n_e_final = n_e0 * np.maximum(1.0 - (r / r[i_wall])**2, 0.0)
    n_e_final[i_wall:] = 0.0   # оболочка: n_e=0

    E_eff_fin  = effective_field(np.sqrt(v), p_pa)
    Da_final   = ambipolar_diffusion(E_eff_fin, p_pa)
    nu_i_final = ionization_freq(E_eff_fin, p_pa)

    # λ₀ вычисляется с тем же ГУ и i_wall, что и в основном решателе
    lambda0_sq = compute_lambda0(r, h, Da_final, nu_i_final,
                                 v_bohm=v_bohm, i_wall=i_wall)
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

    Физика (после исправления формулы λ₀² = μ в compute_lambda0):
    λ₀² = μ = доминирующее собственное значение (−Da·Δ)⁻¹·diag(νi).
    Для однородного случая: λ₀² = νi · R² / (Da · λ₁²).

    - При λ₀ > 1: ионизация избыточна (νi > Da·λ₁²/R²) → плазма растёт.
      Нужно уменьшить n_e0 → усилить скин-эффект → E внутри падает → νi ↓.
    - При λ₀ < 1: ионизация недостаточна (νi < Da·λ₁²/R²) → плазма гаснет.
      Нужно увеличить n_e0.
    - При λ₀ = 1: самосогласованное равновесие.

    Примечание: монотонность λ₀(n_e0) обеспечивается тем, что рост n_e0
    усиливает скин-эффект и уменьшает E_eff внутри → νi падает → λ₀ убывает.
    Бисекция ищет переход λ₀ > 1 → λ₀ < 1.

    Returns dict: n_e0, lambda0, converged, n_bisect, solution, history
    """
    kw_base = dict(max_iter=MAX_ITER, tol=TOL, relax=RELAX)
    if solver_kw is not None:
        kw_base.update(solver_kw)

    log_lo = np.log10(n_e0_bounds[0])
    log_hi = np.log10(n_e0_bounds[1])
    history = []

    # ------------------------------------------------------------------
    # Проверка скобки: λ₀ должна быть по разные стороны от 1 на концах.
    # ------------------------------------------------------------------
    def _lam_effective(res: dict) -> float:
        """λ₀ из результата; 1e10 если не сошлось."""
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

    r_lo = solve_maxwell_for_ne0(n_e0=10.0**log_lo, N=N, R=R, p_pa=p_pa,
                                  H_wall=H_wall, **_adaptive_kw(10.0**log_lo))
    r_hi = solve_maxwell_for_ne0(n_e0=10.0**log_hi, N=N, R=R, p_pa=p_pa,
                                  H_wall=H_wall, **_adaptive_kw(10.0**log_hi))
    lam_lo = _lam_effective(r_lo)
    lam_hi = _lam_effective(r_hi)

    history.append((10.0**log_lo, r_lo["lambda0"]))
    history.append((10.0**log_hi, r_hi["lambda0"]))

    if verbose:
        print(f"  bracket check: n_e0={10.0**log_lo:.3e} lambda0={lam_lo:.4f}"
              f"  |  n_e0={10.0**log_hi:.3e} lambda0={lam_hi:.4f}")

    # Проверяем: не является ли уже граничная точка корнем.
    # Делаем это до проверки скобки, чтобы не потерять точное попадание.
    best_result = None
    best_n_e0   = None
    best_lam0   = None

    for ne0_end, res_end in ((10.0**log_lo, r_lo), (10.0**log_hi, r_hi)):
        if res_end["converged"]:
            if best_result is None or abs(res_end["lambda0"] - 1.0) < abs(best_lam0 - 1.0):
                best_result = res_end
                best_n_e0   = ne0_end
                best_lam0   = res_end["lambda0"]

    # Если граничная точка уже в допуске — возвращаем сразу.
    if best_result is not None and abs(best_lam0 - 1.0) < tol_lambda:
        if verbose:
            print(f"  bracket endpoint is already a root: "
                  f"n_e0={best_n_e0:.3e}  lambda0={best_lam0:.6f}")
        return {
            "n_e0":       best_n_e0,
            "lambda0":    best_lam0,
            "converged":  True,
            "n_bisect":   len(history),
            "solution":   best_result,
            "history":    history,
            "bracket_ok": True,
        }

    # Проверяем скобку: λ₀ должна менять знак (λ₀−1) между концами.
    # Используем <= 0 чтобы включить нулевое произведение (точное λ₀=1
    # на границе уже обработано выше, так что здесь это не достижимо).
    bracket_ok = (lam_lo - 1.0) * (lam_hi - 1.0) <= 0.0
    if not bracket_ok:
        # best_result уже содержит ближайшую к 1 сошедшуюся граничную точку
        # (или None, если обе не сошлись); выбираем лучшее из доступного.
        if best_result is None:
            if abs(lam_lo - 1.0) <= abs(lam_hi - 1.0):
                best_n_e0, best_lam0 = 10.0**log_lo, r_lo["lambda0"]
            else:
                best_n_e0, best_lam0 = 10.0**log_hi, r_hi["lambda0"]
        if verbose:
            print("  WARNING: bracket does not straddle lambda0=1; "
                  "no root in given n_e0 range. Returning nearest endpoint.")
        return {
            "n_e0":       best_n_e0,
            "lambda0":    best_lam0,
            "converged":  False,
            "n_bisect":   len(history),
            "solution":   best_result,
            "history":    history,
            "bracket_ok": False,
        }

    # Скобка валидна. best_result уже инициализирован из граничных точек
    # (лучшей из сошедшихся), так что внутренние точки могут только улучшить его.

    n_e0_trial = 10.0**(0.5*(log_lo + log_hi))
    lam0       = 0.5 * (lam_lo + lam_hi)   # только для инициализации

    for step in range(max_bisect):
        log_mid    = 0.5 * (log_lo + log_hi)
        n_e0_trial = 10.0**log_mid

        result = solve_maxwell_for_ne0(
            n_e0=n_e0_trial, N=N, R=R, p_pa=p_pa, H_wall=H_wall,
            **_adaptive_kw(n_e0_trial))
        lam0          = result["lambda0"]
        lam0_effective = _lam_effective(result)

        history.append((n_e0_trial, lam0))

        if verbose:
            print(f"  bisect {step:3d}: n_e0={n_e0_trial:.3e}  lambda0={lam0:.6f}"
                  f"  conv={result['converged']}  iters={result['n_iter']}")

        # Обновляем best_result только при сходимости Maxwell-решателя.
        # Храним n_e0 и lambda0 из того же прогона — всегда согласованы.
        if result["converged"]:
            if best_result is None or abs(lam0 - 1.0) < abs(best_lam0 - 1.0):
                best_result = result
                best_n_e0   = n_e0_trial
                best_lam0   = lam0

        if result["converged"] and abs(lam0 - 1.0) < tol_lambda:
            break

        # Сдвигаем соответствующий конец скобки
        if (lam_lo - 1.0) * (lam0_effective - 1.0) > 0.0:
            log_lo = log_mid
            lam_lo = lam0_effective
        else:
            log_hi = log_mid
            lam_hi = lam0_effective

    # Если ни один шаг не дал сошедшегося решения, возвращаем последнюю точку
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
