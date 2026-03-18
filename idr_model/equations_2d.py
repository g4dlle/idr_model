"""
equations_2d.py — дискретизация системы уравнений на 2D
осесимметричной сетке (r, z) методом конечных разностей.

Сетка
─────
  r_i = i·hr,   i = 0 .. Nr      (радиальная координата)
  z_j = j·hz,   j = 0 .. Nz      (осевая координата)

Линейная нумерация: k = i*(Nz+1) + j

Уравнения
─────────
(1/r)∂/∂r(r·α·∂u/∂r) + ∂/∂z(α·∂u/∂z) = 2σ_a·v

(1/r)∂/∂r(r·Da·∂σ/∂r) + ∂/∂z(Da·∂σ/∂z) + νi·σ = 0

Матрицы собираются в формате scipy.sparse.csr_matrix.
"""

import numpy as np
from scipy import sparse


# ─────────────────────────────────────────────────────────────────────────────
# Сетка
# ─────────────────────────────────────────────────────────────────────────────

def make_grid_2d(Nr: int, Nz: int, R: float, L: float,
                 r_min: float = 0.0):
    """
    Равномерная 2D-сетка (r, z).

    Parameters
    ----------
    Nr    : число интервалов по r
    Nz    : число интервалов по z
    R     : внешний радиус [м]
    L     : длина трубки по z [м]
    r_min : внутренний радиус [м] (0 — ось, >0 — кольцо/включение)

    Returns
    -------
    r  : 1D-массив (Nr+1,), координаты по r
    z  : 1D-массив (Nz+1,), координаты по z
    hr : шаг по r
    hz : шаг по z
    """
    r = np.linspace(r_min, R, Nr + 1)
    z = np.linspace(0.0, L, Nz + 1)
    hr = (R - r_min) / Nr
    hz = L / Nz
    return r, z, hr, hz


def idx(i: int, j: int, Nz1: int) -> int:
    """Линейная нумерация (i, j) → k.  Nz1 = Nz + 1."""
    return i * Nz1 + j


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение для |H|² = u    [2D]
# ─────────────────────────────────────────────────────────────────────────────

def build_H_equation_2d(r, z, hr, hz,
                        alpha_2d, sigma_a_2d, v_2d,
                        H_wall_sq,
                        bc_z="neumann"):
    """
    Собирает разреженную СЛАУ для уравнения на 2D-сетке.

    (1/r)∂/∂r(r·α·∂u/∂r) + ∂/∂z(α·∂u/∂z) = 2·σ_a·v

    ГУ:
      r = 0   : du/dr = 0 (Лопиталь → 2α·∂²u/∂r²)
      r = R   : u = H_wall² (Дирихле)
      z = 0, L: ∂u/∂z = 0 (Неймана) или Дирихле (настраивается)

    Parameters
    ----------
    r, z       : 1D-сетки
    hr, hz     : шаги
    alpha_2d   : α = σ_a/|σ|²,  shape (Nr+1, Nz+1)
    sigma_a_2d : активная проводимость, shape (Nr+1, Nz+1)
    v_2d       : |E|²,  shape (Nr+1, Nz+1)
    H_wall_sq  : |H|²(R)
    bc_z       : "neumann" или "dirichlet"

    Returns
    -------
    A   : scipy.sparse.csr_matrix,  shape (M, M)
    rhs : numpy array (M,)
    """
    Nr = len(r) - 1
    Nz = len(z) - 1
    Nz1 = Nz + 1
    M = (Nr + 1) * Nz1

    rows = []
    cols = []
    vals = []
    rhs = np.zeros(M)

    def add(row, col, val):
        rows.append(row)
        cols.append(col)
        vals.append(val)

    for i in range(Nr + 1):
        for j in range(Nz1):
            k = idx(i, j, Nz1)

            # ── Дирихле на стенке r = R ──────────────────────────
            if i == Nr:
                add(k, k, 1.0)
                rhs[k] = H_wall_sq
                continue

            # ── ГУ по z ──────────────────────────────────────────
            if bc_z == "dirichlet" and (j == 0 or j == Nz):
                add(k, k, 1.0)
                rhs[k] = H_wall_sq  # или другое значение
                continue

            # ── Ось r = 0: правило Лопиталя ──────────────────────
            if i == 0 and r[0] == 0.0:
                # Радиальная часть: lim_{r→0} (1/r)∂/∂r(r·α·∂u/∂r) = 2α·∂²u/∂r²
                ap = 0.5 * (alpha_2d[0, j] + alpha_2d[1, j])
                coef_r = 2.0 * ap / hr**2

                # z-часть: стандартная
                coef_z_p = 0.0
                coef_z_m = 0.0
                diag_z = 0.0
                if j > 0 and j < Nz:
                    az_p = 0.5 * (alpha_2d[i, j] + alpha_2d[i, j + 1])
                    az_m = 0.5 * (alpha_2d[i, j] + alpha_2d[i, j - 1])
                    coef_z_p = az_p / hz**2
                    coef_z_m = az_m / hz**2
                    diag_z = -(coef_z_p + coef_z_m)
                elif j == 0:
                    # Неймана: du/dz = 0 → u[i,-1] = u[i,1]
                    az_p = 0.5 * (alpha_2d[i, 0] + alpha_2d[i, 1])
                    coef_z_p = 2.0 * az_p / hz**2
                    diag_z = -coef_z_p
                else:  # j == Nz
                    az_m = 0.5 * (alpha_2d[i, Nz] + alpha_2d[i, Nz - 1])
                    coef_z_m = 2.0 * az_m / hz**2
                    diag_z = -coef_z_m

                # Диагональ
                add(k, k, -coef_r + diag_z)
                # u[1, j]
                add(k, idx(1, j, Nz1), coef_r)
                # z-соседи
                if j > 0:
                    add(k, idx(i, j - 1, Nz1), coef_z_m)
                if j < Nz:
                    add(k, idx(i, j + 1, Nz1), coef_z_p)

                rhs[k] = 2.0 * sigma_a_2d[i, j] * v_2d[i, j]
                continue

            # ── Кольцевая ось r[0] > 0: Неймана du/dr = 0 ────────
            if i == 0 and r[0] > 0.0:
                add(k, k, 1.0)
                add(k, idx(1, j, Nz1), -1.0)
                rhs[k] = 0.0
                continue

            # ── Внутренние узлы ───────────────────────────────────
            ri = r[i]
            rp = 0.5 * (r[i] + r[i + 1])  # r_{i+1/2}
            rm = 0.5 * (r[i] + r[i - 1])  # r_{i-1/2}
            ap = 0.5 * (alpha_2d[i, j] + alpha_2d[i + 1, j])
            am = 0.5 * (alpha_2d[i, j] + alpha_2d[i - 1, j])

            # Радиальный оператор: (1/(ri·hr²)) · [rp·ap·(u_{i+1}-u_i) - rm·am·(u_i-u_{i-1})]
            coef_r = 1.0 / (ri * hr**2)
            cr_p = coef_r * rp * ap  # u_{i+1}
            cr_m = coef_r * rm * am  # u_{i-1}
            cr_d = -(cr_p + cr_m)    # u_{i}

            # z-оператор: (1/hz²) · [az_p·(u_{j+1}-u_j) - az_m·(u_j-u_{j-1})]
            coef_z_p = 0.0
            coef_z_m = 0.0
            if j > 0 and j < Nz:
                az_p = 0.5 * (alpha_2d[i, j] + alpha_2d[i, j + 1])
                az_m = 0.5 * (alpha_2d[i, j] + alpha_2d[i, j - 1])
                coef_z_p = az_p / hz**2
                coef_z_m = az_m / hz**2
            elif j == 0:
                # Неймана: ghost → u[i,-1] = u[i,1]
                az_p = 0.5 * (alpha_2d[i, 0] + alpha_2d[i, 1])
                coef_z_p = 2.0 * az_p / hz**2
                coef_z_m = 0.0
            else:  # j == Nz
                az_m = 0.5 * (alpha_2d[i, Nz] + alpha_2d[i, Nz - 1])
                coef_z_p = 0.0
                coef_z_m = 2.0 * az_m / hz**2
            diag_z = -(coef_z_p + coef_z_m)

            # Собираем строку
            add(k, k, cr_d + diag_z)
            add(k, idx(i + 1, j, Nz1), cr_p)
            add(k, idx(i - 1, j, Nz1), cr_m)
            if j < Nz:
                add(k, idx(i, j + 1, Nz1), coef_z_p)
            if j > 0:
                add(k, idx(i, j - 1, Nz1), coef_z_m)

            rhs[k] = 2.0 * sigma_a_2d[i, j] * v_2d[i, j]

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(M, M))
    return A, rhs


# ─────────────────────────────────────────────────────────────────────────────
# Уравнение для σ [2D]
# ─────────────────────────────────────────────────────────────────────────────

def build_sigma_equation_2d(r, z, hr, hz,
                            Da_2d, nu_i_2d,
                            sigma_ref=None, dt=None,
                            bc_z_sigma="dirichlet"):
    """
    Собирает разреженную СЛАУ для уравнения на 2D-сетке.

    (1/r)∂/∂r(r·Da·∂σ/∂r) + ∂/∂z(Da·∂σ/∂z) + νi·σ = 0

    Режим power iteration: (-L)·σ_new = νi·σ_ref

    ГУ:
      r = 0   : dσ/dr = 0 (Лопиталь)
      r = R   : σ = 0
      z = 0, L: σ = 0 (Дирихле) или dσ/dz = 0 (Неймана)

    Parameters
    ----------
    r, z           : 1D-сетки
    hr, hz         : шаги
    Da_2d          : амбиполярный коэффициент диффузии, shape (Nr+1, Nz+1)
    nu_i_2d        : частота ионизации, shape (Nr+1, Nz+1)
    sigma_ref      : профиль σ с предыдущего шага; None → однородная система
    dt             : шаг псевдо-времени; None → без временного члена
    bc_z_sigma     : "dirichlet" или "neumann" — ГУ для σ на торцах

    Returns
    -------
    A   : scipy.sparse.csr_matrix
    rhs : numpy array
    """
    Nr = len(r) - 1
    Nz = len(z) - 1
    Nz1 = Nz + 1
    M = (Nr + 1) * Nz1

    rows = []
    cols = []
    vals = []
    rhs = np.zeros(M)

    power_iter = sigma_ref is not None
    time_step = power_iter and (dt is not None)
    inv_dt = 1.0 / dt if time_step else 0.0

    def add(row, col, val):
        rows.append(row)
        cols.append(col)
        vals.append(val)

    for i in range(Nr + 1):
        for j in range(Nz1):
            k = idx(i, j, Nz1)

            # ── Дирихле σ = 0 на стенке r = R ────────────────────
            if i == Nr:
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue

            # ── Дирихле σ = 0 на торцах z = 0, L ─────────────────
            if bc_z_sigma == "dirichlet" and (j == 0 or j == Nz):
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue

            # ── Ось r = 0: правило Лопиталя ──────────────────────
            if i == 0 and r[0] == 0.0:
                Dp = 0.5 * (Da_2d[0, j] + Da_2d[1, j])
                coef_r = 2.0 * Dp / hr**2

                # z-часть
                coef_z_p = 0.0
                coef_z_m = 0.0
                if j > 0 and j < Nz:
                    Dz_p = 0.5 * (Da_2d[i, j] + Da_2d[i, j + 1])
                    Dz_m = 0.5 * (Da_2d[i, j] + Da_2d[i, j - 1])
                    coef_z_p = Dz_p / hz**2
                    coef_z_m = Dz_m / hz**2
                elif j == 0:
                    if bc_z_sigma == "neumann":
                        Dz_p = 0.5 * (Da_2d[i, 0] + Da_2d[i, 1])
                        coef_z_p = 2.0 * Dz_p / hz**2
                    else:
                        # j=0 — Дирихле, обработано выше
                        pass
                else:  # j == Nz
                    if bc_z_sigma == "neumann":
                        Dz_m = 0.5 * (Da_2d[i, Nz] + Da_2d[i, Nz - 1])
                        coef_z_m = 2.0 * Dz_m / hz**2
                diag_z = -(coef_z_p + coef_z_m)

                if power_iter:
                    add(k, k, coef_r + (-diag_z) + inv_dt)
                    add(k, idx(1, j, Nz1), -coef_r)
                    if j > 0:
                        add(k, idx(i, j - 1, Nz1), -coef_z_m)
                    if j < Nz:
                        add(k, idx(i, j + 1, Nz1), -coef_z_p)
                    if time_step:
                        rhs[k] = sigma_ref[i, j] * (inv_dt + nu_i_2d[i, j])
                    else:
                        rhs[k] = nu_i_2d[i, j] * sigma_ref[i, j]
                else:
                    add(k, k, coef_r + (-diag_z) - nu_i_2d[i, j])
                    add(k, idx(1, j, Nz1), -coef_r)
                    if j > 0:
                        add(k, idx(i, j - 1, Nz1), -coef_z_m)
                    if j < Nz:
                        add(k, idx(i, j + 1, Nz1), -coef_z_p)
                    rhs[k] = 0.0
                continue

            # ── Кольцевая ось r[0] > 0: Дирихле σ = 0 ────────────
            if i == 0 and r[0] > 0.0:
                add(k, k, 1.0)
                rhs[k] = 0.0
                continue

            # ── Внутренние узлы ───────────────────────────────────
            ri = r[i]
            rp = 0.5 * (r[i] + r[i + 1])
            rm = 0.5 * (r[i] + r[i - 1])
            Dp = 0.5 * (Da_2d[i, j] + Da_2d[i + 1, j])
            Dm = 0.5 * (Da_2d[i, j] + Da_2d[i - 1, j])

            coef_r = 1.0 / (ri * hr**2)
            cr_p = coef_r * rp * Dp
            cr_m = coef_r * rm * Dm

            # z-часть
            coef_z_p = 0.0
            coef_z_m = 0.0
            if j > 0 and j < Nz:
                Dz_p = 0.5 * (Da_2d[i, j] + Da_2d[i, j + 1])
                Dz_m = 0.5 * (Da_2d[i, j] + Da_2d[i, j - 1])
                coef_z_p = Dz_p / hz**2
                coef_z_m = Dz_m / hz**2
            elif j == 0:
                if bc_z_sigma == "neumann":
                    Dz_p = 0.5 * (Da_2d[i, 0] + Da_2d[i, 1])
                    coef_z_p = 2.0 * Dz_p / hz**2
            else:  # j == Nz
                if bc_z_sigma == "neumann":
                    Dz_m = 0.5 * (Da_2d[i, Nz] + Da_2d[i, Nz - 1])
                    coef_z_m = 2.0 * Dz_m / hz**2
            diag_z = -(coef_z_p + coef_z_m)

            if power_iter:
                add(k, k, (cr_p + cr_m) + (-diag_z) + inv_dt)
                add(k, idx(i + 1, j, Nz1), -cr_p)
                add(k, idx(i - 1, j, Nz1), -cr_m)
                if j < Nz:
                    add(k, idx(i, j + 1, Nz1), -coef_z_p)
                if j > 0:
                    add(k, idx(i, j - 1, Nz1), -coef_z_m)
                if time_step:
                    rhs[k] = sigma_ref[i, j] * (inv_dt + nu_i_2d[i, j])
                else:
                    rhs[k] = nu_i_2d[i, j] * sigma_ref[i, j]
            else:
                add(k, k, (cr_p + cr_m) + (-diag_z) - nu_i_2d[i, j])
                add(k, idx(i + 1, j, Nz1), -cr_p)
                add(k, idx(i - 1, j, Nz1), -cr_m)
                if j < Nz:
                    add(k, idx(i, j + 1, Nz1), -coef_z_p)
                if j > 0:
                    add(k, idx(i, j - 1, Nz1), -coef_z_m)
                rhs[k] = 0.0

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(M, M))
    return A, rhs
