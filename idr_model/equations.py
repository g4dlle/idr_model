"""
equations.py — дискретизация системы уравнений (11)–(13) методом конечных
разностей на равномерной сетке r_i = i·h, i = 0 .. N.

Система уравнений
─────────────────
(11)  (1/r) d/dr [ r · (σ_a/|σ|²) · d|H|²/dr ] = 2·σ_a·|E|²

(12)  (1/r²) d/dr [ r² · (σ_a/|σ|²) · d|E|²/dr ] = 2·σ_a·|H|²/r²·...
      (полный вид из статьи: d²|E|²/dr² + (2/r)·d|E|²/dr - (2/r²)·|E|² = 2·σ_a·|H|²·... )

(13)  Da·(d²σ/dr² + (1/r)·dσ/dr) + νi·σ = 0

Дискретизация
─────────────
Коэффициенты на полуцелых узлах:

  α_{i+1/2} = (σ_a/|σ|²)_{i+1/2}  ≈  (α_i + α_{i+1})/2

Оператор (1/r_i) · d/dr(r · α · du/dr) на сетке:

  L_i[u] = (1/(r_i · h²)) · [r_{i+1/2}·α_{i+1/2}·(u_{i+1}-u_i)
                              - r_{i-1/2}·α_{i-1/2}·(u_i-u_{i-1})]

Особая точка r=0 (правило Лопиталя):
  lim_{r→0} (1/r)·d/dr(r·f') = 2·f''  →  L_0[u] = 2·α_{1/2}·(u_1-u_0)/h²

Трёхдиагональная СЛАУ: A·x = b (строится по каждому уравнению).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(N: int, R: float, r_min: float = 0.0) -> tuple[np.ndarray, float]:
    """
    Равномерная сетка по r.

    Parameters
    ----------
    N     : число интервалов (узлов N+1)
    R     : внешний радиус [м]
    r_min : внутренний радиус [м] (0 — полный цилиндр, >0 — кольцо)

    Returns
    -------
    r : массив узлов (N+1,), от r_min до R
    h : шаг сетки [м]
    """
    r = np.linspace(r_min, R, N + 1)
    h = (R - r_min) / N
    return r, h


def half_node_avg(arr: np.ndarray) -> np.ndarray:
    """
    Значение на полуцелых узлах: arr_{i+1/2} = (arr_i + arr_{i+1})/2.

    Returns массив длины len(arr)-1.
    """
    return 0.5 * (arr[:-1] + arr[1:])


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение (11) для |H|² = u
# ─────────────────────────────────────────────────────────────────────────────

def build_H_equation(r: np.ndarray, h: float,
                     alpha: np.ndarray,
                     sigma_a: np.ndarray,
                     v: np.ndarray,
                     H_wall_sq: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит трёхдиагональную СЛАУ для уравнения (11):

      (1/r) d/dr [ r·α·du/dr ] = 2·σ_a·v

    где α = σ_a/|σ|², u = |H|², v = |E|².

    Граничные условия:
      r=0: du/dr=0 (симметрия, ghost node)
      r=R: u[N] = H_wall_sq (Дирихле)

    Parameters
    ----------
    r        : сетка (N+1,)
    h        : шаг сетки
    alpha    : σ_a/|σ|² на узлах (N+1,)
    sigma_a  : активная проводимость на узлах (N+1,)
    v        : |E|² на узлах (N+1,)
    H_wall_sq: |H|²(R) = H_wall²

    Returns
    -------
    lower, main, upper, rhs : компоненты трёхдиагональной системы (длина N+1)
    """
    N = len(r) - 1
    lower = np.zeros(N + 1)
    main  = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs   = np.zeros(N + 1)

    alpha_half = half_node_avg(alpha)          # α_{i+1/2}, длина N
    r_half_plus  = 0.5 * (r[:-1] + r[1:])     # r_{i+1/2}
    r_half_minus = np.zeros(N)                 # r_{i-1/2}
    r_half_minus[1:] = 0.5 * (r[:-2] + r[1:-1])
    r_half_minus[0]  = 0.0                     # граница при i=0

    # ── Внутренние узлы i = 1 .. N-1 ────────────────────────────────────────
    for i in range(1, N):
        rp = r_half_plus[i]    # r_{i+1/2}
        rm = r_half_minus[i]   # r_{i-1/2}
        ap = alpha_half[i]     # α_{i+1/2}
        am = alpha_half[i - 1] # α_{i-1/2}
        ri = r[i]

        coef = 1.0 / (ri * h**2)
        # Уравнение (11): L[u] = 2σ_a·v, L = (1/r)d/dr[r·α·d/dr]
        # Матрица оператора +L: отрицательный диагональ, положительные внедиагональные.
        lower[i] =  coef * rm * am
        upper[i] =  coef * rp * ap
        main[i]  = -coef * (rp * ap + rm * am)
        rhs[i]   = 2.0 * sigma_a[i] * v[i]

    # ── Узел i = 0 ─────────────────────────────────────────────────────────────
    if r[0] > 0:
        # Кольцевая геометрия: r[0] = r_inc > 0.
        # ГУ Неймана du/dr = 0 на внутренней стенке:
        # Одностороння разность: (u[1] - u[0])/h = 0  → u[0] = u[1]
        lower[0] = 0.0
        main[0]  = 1.0
        upper[0] = -1.0
        rhs[0]   = 0.0
    else:
        # Особая точка r=0: правило Лопиталя
        # lim_{r→0} (1/r)·d/dr(r·α·u') = 2·α_{1/2}·(u_1 - u_0)/h²
        ap0 = alpha_half[0]
        main[0]  = -2.0 * ap0 / h**2
        upper[0] =  2.0 * ap0 / h**2
        rhs[0]   =  2.0 * sigma_a[0] * v[0]
        lower[0] = 0.0   # ghost node: u[-1]=u[1]

    # ── Граничное условие r=R (Дирихле) ──────────────────────────────────────
    lower[N] = 0.0
    main[N]  = 1.0
    upper[N] = 0.0
    rhs[N]   = H_wall_sq

    return lower, main, upper, rhs


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение (12) для |E|² = v
# ─────────────────────────────────────────────────────────────────────────────

def build_E_equation(r: np.ndarray, h: float,
                     alpha: np.ndarray,
                     sigma_a: np.ndarray,
                     u: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит трёхдиагональную СЛАУ для уравнения (12):

      (1/r²) d/dr [ r²·α·dv/dr ] - 2·α·v/r² = 2·σ_a·u·...

    Полный вид (уравнение индукции для E в цилиндре):
      d²v/dr² + (2/r)·dv/dr - (2/r²)·v = 2·σ_a·u  (α нормировано)

    То есть оператор: α·[ v'' + (2/r)·v' - 2/r²·v ] = RHS

    Граничные условия:
      r=0:  v[0] = 0  (Дирихле, E=0 на оси)
      r=R:  dv/dr + v/R = 0  (условие сопряжения, упрощённо — Неймана)
            или задаётся внешним полем; здесь используем Неймана нулевого порядка.

    Parameters
    ----------
    r       : сетка (N+1,)
    h       : шаг сетки
    alpha   : σ_a/|σ|² на узлах (N+1,)
    sigma_a : активная проводимость (N+1,)
    u       : |H|² на узлах (N+1,)

    Returns
    -------
    lower, main, upper, rhs
    """
    N = len(r) - 1
    lower = np.zeros(N + 1)
    main  = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs   = np.zeros(N + 1)

    alpha_half = half_node_avg(alpha)

    # ── Граничное условие i=0: Дирихле v=0 ────────────────────────────────────
    # При r[0]=0 (ось): E_φ=0 по симметрии.
    # При r[0]>0 (включение): E_φ=0 на проводнике.
    # В обоих случаях: v[0] = 0.
    main[0]  = 1.0
    upper[0] = 0.0
    rhs[0]   = 0.0

    # ── Внутренние узлы i = 1 .. N-1 ────────────────────────────────────────
    # Дискретизация (1/r²)·d/dr(r²·α·dv/dr) - 2·α·v/r²
    # = α·[ (v_{i+1}-v_i)·r²_{i+1/2} - (v_i-v_{i-1})·r²_{i-1/2} ] / (r_i²·h²)
    #   - 2·α_i·v_i/r_i²
    r_sq_half_plus  = (0.5*(r[:-1] + r[1:]))**2    # r²_{i+1/2}
    r_sq_half_minus = np.zeros(N)
    r_sq_half_minus[1:] = (0.5*(r[:-2] + r[1:-1]))**2

    for i in range(1, N):
        rsp = r_sq_half_plus[i]
        rsm = r_sq_half_minus[i]
        ap  = alpha_half[i]
        am  = alpha_half[i - 1]
        ri2 = r[i]**2

        coef = 1.0 / (ri2 * h**2)
        lower[i] = -coef * rsm * am
        upper[i] = -coef * rsp * ap
        main[i]  =  coef * (rsp * ap + rsm * am) + 2.0 * alpha[i] / ri2
        rhs[i]   = 2.0 * sigma_a[i] * u[i]

    # ── Граничное условие r=R: dv/dr = 0 (Неймана) ───────────────────────────
    # Одностороннее: (v[N] - v[N-1])/h = 0  → v[N] = v[N-1]
    lower[N] = -1.0
    main[N]  =  1.0
    upper[N] =  0.0
    rhs[N]   =  0.0

    return lower, main, upper, rhs


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение (13) для σ
# ─────────────────────────────────────────────────────────────────────────────

def build_sigma_equation(r: np.ndarray, h: float,
                          Da: np.ndarray,
                          nu_i: np.ndarray,
                          sigma_a_ref: np.ndarray | None = None,
                          dt: float | None = None,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит трёхдиагональную СЛАУ для уравнения (13):

      (1/r)·d/dr(r·Da·dσ/dr) + νi·σ = 0

    Три режима (управляются параметрами sigma_a_ref и dt):

    • sigma_a_ref is None, dt is None — однородная система (классическая форма):
        A·σ = 0,  где  A_{ii} = ... − νi[i]
      Нетривиальное решение существует только при νi/Da = собственному значению.
      Используется в тестах.

    • sigma_a_ref is not None, dt is None — итерация по мощности (power iteration):
        (−Da·Δ)·σ_new = νi·σ_ref
      Сходится к фундаментальной моде J₀, но амплитуда зафиксирована через
      нормировку вне этой функции.

    • sigma_a_ref is not None, dt is not None — псевдоустановление по времени (IMEX):
        (σ_new − σ_old)/dt = (1/r)·d/dr(r·Da·dσ_new/dr) + νi·σ_old
      Диффузия неявная (устойчиво), ионизация явная.
      При сходимости σ_new → σ_old, что соответствует физическому равновесию
        (1/r)·d/dr(r·Da·dσ/dr) + νi·σ = 0.
      Амплитуда эволюционирует самосогласованно до баланса νi = λ₁·Da.

    Граничные условия:
      r=0: dσ/dr = 0  (ghost node / правило Лопиталя)
      r=R: σ[N]  = 0  (Дирихле, рекомбинация на стенке)

    Parameters
    ----------
    r          : сетка (N+1,)
    h          : шаг сетки [м]
    Da         : амбиполярный коэффициент диффузии (N+1,) [м²/с]
    nu_i       : частота ионизации (N+1,) [с⁻¹]
    sigma_a_ref: профиль σ с предыдущего шага (N+1,); None → однородная форма
    dt         : шаг псевдо-времени [с]; None → без временного члена

    Returns
    -------
    lower, main, upper, rhs : компоненты трёхдиагональной системы (N+1,)
    """
    N = len(r) - 1
    lower = np.zeros(N + 1)
    main  = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs   = np.zeros(N + 1)

    Da_half = half_node_avg(Da)
    r_half_plus  = 0.5 * (r[:-1] + r[1:])
    r_half_minus = np.zeros(N)
    r_half_minus[1:] = 0.5*(r[:-2] + r[1:-1])

    power_iter  = sigma_a_ref is not None
    time_step   = power_iter and (dt is not None)
    inv_dt      = 1.0 / dt if time_step else 0.0

    # ── Узел i = 0 ────────────────────────────────────────────────────────────
    if r[0] > 0:
        # Кольцевая геометрия: σ(r_inc) = 0 (Дирихле — рекомбинация)
        lower[0] = 0.0
        main[0]  = 1.0
        upper[0] = 0.0
        rhs[0]   = 0.0
    else:
        # Особая точка r=0: правило Лопиталя
        Dp0 = Da_half[0]
        if power_iter:
            main[0]  =  2.0 * Dp0 / h**2 + inv_dt
            if time_step:
                rhs[0] = sigma_a_ref[0] * (inv_dt + nu_i[0])
            else:
                rhs[0] = nu_i[0] * sigma_a_ref[0]
        else:
            main[0]  =  2.0 * Dp0 / h**2 - nu_i[0]
            rhs[0]   = 0.0
        upper[0] = -2.0 * Dp0 / h**2

    # ── Внутренние узлы i = 1 .. N-1 ─────────────────────────────────────────
    for i in range(1, N):
        rp = r_half_plus[i]
        rm = r_half_minus[i]
        Dp = Da_half[i]
        Dm = Da_half[i - 1]
        ri = r[i]

        coef = 1.0 / (ri * h**2)
        lower[i] = -coef * rm * Dm
        upper[i] = -coef * rp * Dp
        if power_iter:
            main[i] = coef * (rp * Dp + rm * Dm) + inv_dt
            if time_step:
                rhs[i] = sigma_a_ref[i] * (inv_dt + nu_i[i])
            else:
                rhs[i] = nu_i[i] * sigma_a_ref[i]
        else:
            main[i]  =  coef * (rp * Dp + rm * Dm) - nu_i[i]
            rhs[i]   = 0.0

    # ── Граничное условие r=R (Дирихле σ=0) ──────────────────────────────────
    lower[N] = 0.0
    main[N]  = 1.0
    upper[N] = 0.0
    rhs[N]   = 0.0

    return lower, main, upper, rhs
