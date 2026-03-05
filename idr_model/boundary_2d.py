"""
boundary_2d.py — граничные условия для 2D осесимметричной модели ИДР.

Все функции работают с 2D-массивами shape (Nr+1, Nz+1).
Индексирование: arr[i, j], i — по r, j — по z.
"""

import numpy as np


def apply_bc_H_2d(u: np.ndarray, H_wall_sq: float,
                  bc_z: str = "neumann") -> None:
    """
    Граничные условия для |H|² на 2D-сетке.

    r = R (i = -1): u = H_wall²  (Дирихле)
    r = 0 (i = 0) : du/dr = 0    (обрабатывается в equations_2d)
    z-границы      : Неймана или Дирихле
    """
    # Стенка r = R
    u[-1, :] = H_wall_sq
    # z-границы
    if bc_z == "neumann":
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]


def apply_bc_E_2d(v: np.ndarray, bc_z: str = "neumann") -> None:
    """
    ГУ для |E|² на 2D-сетке.

    r = 0 (i = 0): v = 0  (Дирихле: E_φ = 0 на оси)
    r = R         : обрабатывается в equations / integral
    z-границы     : Неймана
    """
    v[0, :] = 0.0
    if bc_z == "neumann":
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]


def apply_bc_sigma_2d(sigma: np.ndarray,
                      bc_z: str = "dirichlet") -> None:
    """
    ГУ для σ на 2D-сетке.

    r = 0 (i = 0) : dσ/dr = 0   (обрабатывается в equations_2d)
    r = R (i = -1): σ = 0       (Дирихле — рекомбинация)
    z = 0, L      : σ = 0 (Дирихле) или dσ/dz = 0 (Неймана)
    """
    sigma[-1, :] = 0.0
    if bc_z == "dirichlet":
        sigma[:, 0] = 0.0
        sigma[:, -1] = 0.0
    else:
        sigma[:, 0] = sigma[:, 1]
        sigma[:, -1] = sigma[:, -2]
