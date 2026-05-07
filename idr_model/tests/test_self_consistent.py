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
    При Da=1, νi=1, R=1: λ₀² = μ = νi·R²/(Da·λ₁²) = 1/λ₁² ≈ 0.1729.

    Физика: степенная итерация сходится к наибольшему собственному значению
    оператора (−Da·Δ)⁻¹·diag(νi), которое равно νi·R²/(Da·λ₁²).
    При νi = Da·(λ₁/R)² получаем λ₀ = 1 (самоподдерживающийся разряд).
    При νi = 1 < Da·λ₁² = 5.783: разряд ниже порога → λ₀ < 1.
    """
    from self_consistent import compute_lambda0

    N = 200
    R = 1.0
    r, h = make_grid(N, R)

    Da   = np.ones(N + 1)
    nu_i = np.ones(N + 1)

    lam0_sq = compute_lambda0(r, h, Da, nu_i)

    j0_root1 = 2.4048255577
    # Правильная формула: λ₀² = νi·R²/(Da·λ₁²) = 1/λ₁²
    expected = 1.0 / j0_root1**2   # ≈ 0.1729

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
      λ₀² = νi·R² / (Da·j₀₁²)

    Физика: λ₀² — наибольшее собственное значение оператора (−Da·Δ)⁻¹·diag(νi).
    Для однородных коэффициентов: λ₀² = νi / (Da·(λ₁/R)²) = νi·R² / (Da·λ₁²).
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
    expected = nu_val * R**2 / (D_val * j0_root1**2)

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
    эффективная ионизация падает → λ₀ убывает (меньше ионизации → меньше λ₀).
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

    # Физика: рост n_e0 → усиление скин-эффекта → E уменьшается внутри
    # → νi падает → λ₀ убывает. Проверяем строгое убывание.
    diffs = [lambdas[i+1] - lambdas[i] for i in range(len(lambdas)-1)]
    all_decreasing = all(d < 0 for d in diffs)

    assert all_decreasing, (
        f"λ₀(n_e0) не убывает с ростом n_e0: "
        f"ne={ne_values}, λ₀={[f'{l:.4f}' for l in lambdas]}"
    )


# ---------------------------------------------------------------------------
# 5. find_n_e0: бисекция прогрессирует к λ₀=1
# ---------------------------------------------------------------------------
def test_find_ne0_converges(monkeypatch):
    """
    Алгоритм бисекции по n_e0:
      - λ₀(n_e0) монотонно убывает с ростом n_e0 (исправленная формула λ₀²=μ):
        при малом n_e0 → слабый скин-эффект → E велико → νi высоко → λ₀ > 1;
        при большом n_e0 → сильный скин-эффект → E мало → νi низко → λ₀ < 1.
      - Алгоритм должен выполнить не менее 3 шагов бисекции.
      - |λ₀* - 1| меньше, чем |λ₀(lo) - 1|  (прогресс к 1).

    Используем monkeypatch для изоляции логики бисекции от физической модели.
    Монотонная функция λ₀(n_e0) убывает от 3.0 до 0.3 на диапазоне [1e14, 1e22].
    """
    import self_consistent as sc

    # Монотонно убывающая λ₀(n_e0): 3.0 → 0.3 в log-пространстве.
    def fake_solve(n_e0, **kwargs):
        log_lo, log_hi = 14.0, 22.0
        log_ne = np.log10(max(n_e0, 1e-300))
        t = (log_ne - log_lo) / (log_hi - log_lo)   # t ∈ [0, 1]
        lam0 = 3.0 * (0.3 / 3.0) ** t               # 3.0 → 0.3
        base = {
            "r": np.array([0.0, 0.006, 0.012]),
            "u": np.array([1e10, 5e9, 1e10]),
            "v": np.array([0.0, 1e4, 2e4]),
            "sigma_a": np.array([1.0, 0.5, 0.0]),
            "sigma_p": np.array([0.1, 0.05, 0.0]),
            "n_e": np.array([1e16, 5e15, 0.0]),
            "Da": np.array([1.0, 1.0, 1.0]),
            "nu_i": np.array([1.0, 1.0, 1.0]),
            "n_iter": 10,
            "residuals": [1e-6],
        }
        return {**base, "lambda0": lam0, "converged": True}

    monkeypatch.setattr(sc, "solve_maxwell_for_ne0", fake_solve)

    result = sc.find_n_e0(
        n_e0_bounds=(1e14, 1e22),
        tol_lambda=0.05,
        max_bisect=30,
    )

    # 1. n_e0* положительно
    assert result["n_e0"] > 0, f"n_e0 = {result['n_e0']:.3e} ≤ 0"

    # 2. Бисекция выполнила не менее 3 шагов (2 граничных + хоть 1 внутренний)
    assert result["n_bisect"] >= 3, (
        f"Слишком мало шагов бисекции: {result['n_bisect']}"
    )

    # 3. Лучшее достигнутое λ₀ ближе к 1, чем нижняя граница (λ₀=3.0 → далеко)
    lam0_lo = next(
        (lam for ne, lam in result["history"] if ne == 1e14),
        3.0,
    )
    best_lam0 = result["lambda0"]
    assert abs(best_lam0 - 1.0) < abs(lam0_lo - 1.0), (
        f"Прогресс не достигнут: λ₀* = {best_lam0:.4f}, "
        f"λ₀(lo) = {lam0_lo:.4f}, |λ₀*-1| должно быть < |λ₀(lo)-1|"
    )

    # 4. Алгоритм сошёлся (нашёл λ₀ ≈ 1 в допуске 0.05)
    assert result["converged"], (
        f"Ожидалась сходимость при монотонной λ₀(n_e0), "
        f"получено λ₀={best_lam0:.4f}, converged={result['converged']}"
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


# ---------------------------------------------------------------------------
# 7. Точное попадание λ₀=1 на граничной точке → converged=True
# ---------------------------------------------------------------------------
def test_bracket_endpoint_exact_root(monkeypatch):
    """
    Если λ₀(n_e0_lo) == 1.0 точно, find_n_e0 должна вернуть
    converged=True и bracket_ok=True, а не bracket_ok=False.

    Было: (lam_lo - 1) * (lam_hi - 1) == 0 не удовлетворяло строгому < 0,
    что приводило к ранней ошибочной сигнатуре converged=False.
    """
    import self_consistent as sc

    call_count = {"n": 0}

    def fake_solve(n_e0, **kwargs):
        """
        Первый вызов (нижняя граница) возвращает λ₀=1.0 (точный корень).
        Второй (верхняя граница) — λ₀=10.0.
        Все остальные — λ₀=5.0, несошедшиеся.
        """
        call_count["n"] += 1
        base = {
            "r": np.array([0.0, 0.006, 0.012]),
            "u": np.array([1e10, 5e9, 1e10]),
            "v": np.array([0.0, 1e4, 2e4]),
            "sigma_a": np.array([1.0, 0.5, 0.0]),
            "sigma_p": np.array([0.1, 0.05, 0.0]),
            "n_e": np.array([1e16, 5e15, 0.0]),
            "Da": np.array([1.0, 1.0, 1.0]),
            "nu_i": np.array([1.0, 1.0, 1.0]),
            "n_iter": 10,
            "residuals": [1e-6],
        }
        if call_count["n"] == 1:
            return {**base, "lambda0": 1.0, "converged": True}
        if call_count["n"] == 2:
            return {**base, "lambda0": 10.0, "converged": True}
        return {**base, "lambda0": 5.0, "converged": False}

    monkeypatch.setattr(sc, "solve_maxwell_for_ne0", fake_solve)

    result = sc.find_n_e0(
        n_e0_bounds=(1e14, 1e22),
        tol_lambda=0.05,
        max_bisect=20,
    )

    assert result["converged"], (
        f"Ожидалось converged=True при λ₀=1 на нижней границе, "
        f"получено: converged={result['converged']}, bracket_ok={result.get('bracket_ok')}"
    )
    assert result.get("bracket_ok"), (
        "bracket_ok должен быть True при точном попадании в корень"
    )
    assert result["solution"] is not None, "solution не должен быть None"
    assert abs(result["lambda0"] - 1.0) < 0.05, (
        f"lambda0={result['lambda0']} далеко от 1"
    )


# ---------------------------------------------------------------------------
# 8. Если все внутренние точки не сходятся — solution берётся из границ
# ---------------------------------------------------------------------------
def test_bracket_fallback_uses_boundary_solution(monkeypatch):
    """
    При корректной скобке, но полном провале сходимости внутренних точек
    бисекции, find_n_e0 должна вернуть solution из лучшей граничной точки,
    а не None (что приводило бы к E_R=0 в solve_self_consistent).

    Было: best_result инициализировался только при попадании в tol_lambda;
    при провале всех внутренних точек возвращался solution=None.
    """
    import self_consistent as sc

    call_count = {"n": 0}

    def fake_solve(n_e0, **kwargs):
        call_count["n"] += 1
        base = {
            "r": np.array([0.0, 0.006, 0.012]),
            "u": np.array([1e10, 5e9, 1e10]),
            "v": np.array([0.0, 1e4, 2e4]),
            "sigma_a": np.array([1.0, 0.5, 0.0]),
            "sigma_p": np.array([0.1, 0.05, 0.0]),
            "n_e": np.array([1e16, 5e15, 0.0]),
            "Da": np.array([1.0, 1.0, 1.0]),
            "nu_i": np.array([1.0, 1.0, 1.0]),
            "n_iter": 10,
            "residuals": [1e-6],
        }
        if call_count["n"] == 1:   # нижняя граница: λ₀=0.5, сошлась
            return {**base, "lambda0": 0.5, "converged": True}
        if call_count["n"] == 2:   # верхняя граница: λ₀=5.0, сошлась
            return {**base, "lambda0": 5.0, "converged": True}
        # все внутренние точки — не сошлись
        return {**base, "lambda0": 2.0, "converged": False}

    monkeypatch.setattr(sc, "solve_maxwell_for_ne0", fake_solve)

    result = sc.find_n_e0(
        n_e0_bounds=(1e14, 1e22),
        tol_lambda=0.05,
        max_bisect=10,
    )

    assert result["solution"] is not None, (
        "solution не должен быть None при наличии сошедшихся граничных точек"
    )
    # Ближайшая к 1 среди границ — нижняя (|0.5-1|=0.5 < |5-1|=4)
    assert abs(result["lambda0"] - 0.5) < 1e-9, (
        f"Ожидался lambda0=0.5 (ближайший к 1 из границ), получен {result['lambda0']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
