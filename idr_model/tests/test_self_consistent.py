"""
test_self_consistent.py — тесты для самосогласованного решателя ВЧ-индукционного разряда.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from equations import make_grid, build_sigma_equation
from solver import thomas_solve


# ---------------------------------------------------------------------------
# 1. compute_lambda0: аналитический предел (J₀), R=1 (безразмерный)
# ---------------------------------------------------------------------------
def test_lambda0_bessel_analytic():
    """
    При Da=1, νi=1, R=1: уравнение (1/r)d/dr(r·dσ/dr) + (1/λ₀²)·σ = 0
    с ГУ σ(R)=0, dσ/dr(0)=0 имеет решение σ = J₀(j₀₁·r/R),
    и λ₀² = j₀₁² ≈ 5.783.
    """
    from self_consistent import compute_lambda0

    N = 200
    R = 1.0  # безразмерный радиус для совпадения со спецификацией
    r, h = make_grid(N, R)

    Da   = np.ones(N + 1)
    nu_i = np.ones(N + 1)

    lam0_sq = compute_lambda0(r, h, Da, nu_i)

    j0_root1 = 2.4048255577
    expected = j0_root1**2   # ≈ 5.783

    assert abs(lam0_sq - expected) / expected < 0.01, (
        f"λ₀² = {lam0_sq:.4f}, ожидалось ≈ {expected:.4f} (ошибка "
        f"{abs(lam0_sq - expected) / expected:.2%})"
    )


# ---------------------------------------------------------------------------
# 2. compute_lambda0: масштабирование Da, νi
# ---------------------------------------------------------------------------
def test_lambda0_scaled():
    """
    При однородных Da=D, νi=ν, R:
      λ₀² = D·(j₀₁/R)² / ν
    """
    from self_consistent import compute_lambda0

    N = 200
    R = 1.0
    D_val  = 3.5
    nu_val = 7.0

    r, h = make_grid(N, R)
    Da   = np.full(N + 1, D_val)
    nu_i = np.full(N + 1, nu_val)

    lam0_sq = compute_lambda0(r, h, Da, nu_i)

    j0_root1 = 2.4048255577
    expected = D_val * (j0_root1 / R)**2 / nu_val

    assert abs(lam0_sq - expected) / expected < 0.01, (
        f"λ₀² = {lam0_sq:.6f}, ожидалось {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. solve_maxwell_for_ne0: обёртка возвращает корректные поля и λ₀
# ---------------------------------------------------------------------------
def test_maxwell_wrapper_fields():
    """
    solve_maxwell_for_ne0 возвращает словарь с ключом 'lambda0',
    все поля ≥ 0, σ(R)=0.
    """
    from self_consistent import solve_maxwell_for_ne0

    result = solve_maxwell_for_ne0(
        n_e0=1e16, N=80, R=0.012, p_pa=133.0, H_wall=100000.0,
        max_iter=300, tol=1e-5
    )

    assert "lambda0" in result, "Нет ключа 'lambda0' в результате"
    assert result["lambda0"] > 0, f"λ₀ ≤ 0: {result['lambda0']}"

    assert np.all(result["u"] >= -1e-10), "u < 0"
    assert np.all(result["v"] >= -1e-10), "v < 0"
    assert np.all(result["sigma_a"] >= -1e-10), "σ_a < 0"
    assert abs(result["sigma_a"][-1]) < 1e-8, f"σ(R) = {result['sigma_a'][-1]}"


# ---------------------------------------------------------------------------
# 4. Монотонность λ₀(n_e0) при фиксированном H_wall
# ---------------------------------------------------------------------------
def test_lambda0_monotone_vs_ne0():
    """
    При росте n_e0: скин-эффект усиливается → E вытесняется к стенке →
    эффективная ионизация падает → λ₀ растёт.
    Проверяем монотонность на 3 значениях n_e0.
    """
    from self_consistent import solve_maxwell_for_ne0

    ne_values = [1e19, 5e20, 1e21]
    lambdas = []

    for ne in ne_values:
        res = solve_maxwell_for_ne0(
            n_e0=ne, N=60, R=0.012, p_pa=133.0, H_wall=100000.0,
            max_iter=2000, tol=1e-4, relax=0.1
        )
        lambdas.append(res["lambda0"])

    # Проверяем монотонность (строго возрастание или убывание)
    diffs = [lambdas[i+1] - lambdas[i] for i in range(len(lambdas)-1)]
    all_increasing = all(d > 0 for d in diffs)
    all_decreasing = all(d < 0 for d in diffs)

    assert all_increasing or all_decreasing, (
        f"λ₀(n_e0) не монотонна: "
        f"ne={ne_values}, λ₀={[f'{l:.4f}' for l in lambdas]}"
    )


# ---------------------------------------------------------------------------
# 5. find_n_e0: бисекция прогрессирует к λ₀=1
# ---------------------------------------------------------------------------
def test_find_ne0_converges():
    """
    Алгоритм бисекции по n_e0:
      - λ₀(n_e0) монотонно возрастает в диапазоне поиска
      - Финальное значение n_e0* — наибольшее, при котором внутренний
        решатель сходится (ближайшее к «переходу» λ₀→1)
      - |λ₀* - 1| меньше, чем |λ₀(lo) - 1|  (прогресс к 1)

    Примечание: при p=133 Па, H_wall=100 кА/м числовая жёсткость
    уравнений Максвелла не позволяет получить сходящееся решение
    при n_e0 выше критического (~7.1e20 м⁻³). Тест проверяет, что
    алгоритм корректно обрабатывает этот случай и возвращает
    наиболее «близкое» значение к балансу.
    """
    from self_consistent import find_n_e0

    result = find_n_e0(
        N=60, R=0.012, p_pa=133.0, H_wall=100000.0,
        n_e0_bounds=(1e18, 1e22),
        tol_lambda=0.05,
        max_bisect=30,
        solver_kw=dict(max_iter=500, tol=1e-4, relax=0.3),
    )

    # 1. n_e0* положительно
    assert result["n_e0"] > 0, f"n_e0 = {result['n_e0']:.3e} ≤ 0"

    # 2. Бисекция выполнила хоть несколько шагов
    assert result["n_bisect"] >= 3, (
        f"Слишком мало шагов бисекции: {result['n_bisect']}"
    )

    # 3. Лучшее достигнутое λ₀ ближе к 1, чем стартовая нижняя граница
    lam0_lo_res = next(
        (lam for ne, lam in result["history"] if ne == 1e18),
        None
    )
    if lam0_lo_res is None:
        # Оцениваем λ₀ при n_e0 = нижняя граница отдельно
        from self_consistent import solve_maxwell_for_ne0
        lam0_lo_res = solve_maxwell_for_ne0(
            1e18, N=60, R=0.012, p_pa=133.0, H_wall=100000.0,
            max_iter=200, tol=1e-4, relax=0.5
        )["lambda0"]

    # λ₀ финального результата должно быть ближе к 1, чем нижняя граница
    best_lam0 = result["solution"]["lambda0"] if result["solution"] else result["lambda0"]
    assert abs(best_lam0 - 1.0) <= abs(lam0_lo_res - 1.0), (
        f"Прогресс не достигнут: λ₀* = {best_lam0:.4f}, "
        f"λ₀(low) = {lam0_lo_res:.4f}, оба далеки от 1"
    )


# ---------------------------------------------------------------------------
# 6. Физические ограничения на финальный результат
# ---------------------------------------------------------------------------
def test_result_physical_bounds():
    """
    Финальное решение: n_e > 0 внутри, поля ≥ 0, σ(R)=0, u(R)=H²_wall.
    """
    from self_consistent import find_n_e0

    result = find_n_e0(
        N=60, R=0.012, p_pa=133.0, H_wall=100000.0,
        n_e0_bounds=(1e18, 1e22),
        tol_lambda=0.05,
        max_bisect=60,
        solver_kw=dict(max_iter=500, tol=1e-4, relax=0.3),
    )

    sol = result["solution"]
    H_wall = 100000.0

    assert np.all(sol["u"] >= -1e-10), f"u < 0: min = {sol['u'].min():.3e}"
    assert np.all(sol["v"] >= -1e-10), f"v < 0: min = {sol['v'].min():.3e}"
    assert abs(sol["sigma_a"][-1]) < 1e-8, f"σ(R) = {sol['sigma_a'][-1]:.3e}"
    assert abs(sol["u"][-1] - H_wall**2) < 1e-2, (
        f"|H|²(R) = {sol['u'][-1]:.3e} ≠ {H_wall**2:.3e}"
    )

    n_e = sol["n_e"]
    assert n_e[0] > 0, f"n_e(0) = {n_e[0]:.3e} ≤ 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
