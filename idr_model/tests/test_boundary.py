"""
test_boundary.py — тесты модуля boundary.py.

Тесты 3.1–3.4 из плана реализации.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from boundary import (
    apply_axis_E, apply_wall_H, apply_wall_sigma,
    apply_all_boundary_conditions, get_ghost_node_value,
)
from equations import make_grid, build_H_equation, build_sigma_equation
from solver import thomas_solve
from config import H_WALL


# ─── 3.1 Ghost node для симметрии на оси ─────────────────────────────────────
def test_axis_symmetry_ghost_node():
    """
    Для условия du/dr|_{r=0} = 0: ghost node u[-1] = u[1].

    get_ghost_node_value(arr) должна возвращать arr[1].
    """
    arr = np.array([5.0, 7.0, 9.0, 11.0, 0.0])
    ghost = get_ghost_node_value(arr)
    assert ghost == arr[1], f"ghost={ghost}, expect {arr[1]}"

    # После решения уравнения с симметричным ГУ профиль должен быть симметричен
    N = 50
    R = 0.01
    r, h = make_grid(N, R)

    alpha   = np.ones(N + 1)
    sigma_a = np.ones(N + 1) * 0.1
    v       = np.ones(N + 1) * 0.5
    H_wall_sq = 1.0

    l, m, up, rhs = build_H_equation(r, h, alpha, sigma_a, v, H_wall_sq)
    u = thomas_solve(l, m, up, rhs)

    # Производная на оси (конечно-разностная) должна быть ≈ 0
    du_axis = abs(u[1] - u[0]) / (r[1] - r[0])
    assert du_axis < 1e-3 * abs(u[0]) or du_axis < 1e-6, (
        f"du/dr|_axis = {du_axis:.3e} — нарушена симметрия"
    )


# ─── 3.2 |E|² = 0 на оси ─────────────────────────────────────────────────────
def test_E_zero_at_axis():
    """
    apply_axis_E устанавливает v[0] = 0.
    """
    v = np.array([3.14, 1.0, 2.0, 3.0])
    apply_axis_E(v)
    assert v[0] == 0.0, f"v[0]={v[0]}, ожидалось 0.0"

    # Исходный массив не изменился дальше первого элемента
    assert v[1] == 1.0 and v[2] == 2.0


# ─── 3.3 |H|² = H_wall² на стенке ───────────────────────────────────────────
def test_dirichlet_H_at_wall():
    """
    apply_wall_H устанавливает u[-1] = H_wall².
    """
    u = np.zeros(10)
    H_test = 5.0
    apply_wall_H(u, H_test)
    assert u[-1] == H_test**2, f"u[-1]={u[-1]}, ожидалось {H_test**2}"

    # Используем значение по умолчанию
    u2 = np.zeros(10)
    apply_wall_H(u2)
    assert u2[-1] == H_WALL**2


# ─── 3.4 σ = 0 на стенке ─────────────────────────────────────────────────────
def test_sigma_zero_at_wall():
    """
    apply_wall_sigma устанавливает sigma[-1] = 0.
    """
    sigma = np.ones(20) * 3.0
    apply_wall_sigma(sigma)
    assert sigma[-1] == 0.0, f"sigma[-1]={sigma[-1]}, ожидалось 0.0"

    # Остальные значения не изменились
    assert np.all(sigma[:-1] == 3.0)


# ─── 3.5 apply_all_boundary_conditions применяет все ГУ ──────────────────────
def test_apply_all_boundary_conditions():
    """
    Комплексный тест: после вызова apply_all все ГУ выполнены.
    """
    N = 30
    u = np.ones(N + 1) * 5.0
    v = np.ones(N + 1) * 3.0
    sigma = np.ones(N + 1) * 2.0
    H_test = 7.0

    apply_all_boundary_conditions(u, v, sigma, H_test)

    assert v[0]     == 0.0,        f"v[0]={v[0]}"
    assert u[-1]    == H_test**2,  f"u[-1]={u[-1]}"
    assert sigma[-1] == 0.0,       f"sigma[-1]={sigma[-1]}"


# ─── 3.6 ГУ сохраняют σ(R)=0 ────────────────────────────────
def test_sigma_bc_in_equation():
    """
    После решения с Da=const, νi=0
    граничное условие σ(R)=0 должно быть выполнено точно.
    """
    N = 50
    R = 0.01
    r, h = make_grid(N, R)
    Da   = np.ones(N + 1)
    nu_i = np.zeros(N + 1)

    l, m, up, rhs = build_sigma_equation(r, h, Da, nu_i)
    sigma = thomas_solve(l, m, up, rhs)

    assert abs(sigma[-1]) < 1e-12, f"σ(R)={sigma[-1]:.3e}, ожидалось 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
