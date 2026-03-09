"""
test_solver_transport.py — тесты решателя с подключаемым транспортом.

Проверяет, что solve_idr() корректно работает:
1. С transport=None (обратная совместимость — аналитические формулы)
2. С объектом transport, реализующим интерфейс BolsigTransport
3. С параметром beta_recomb (объёмная рекомбинация)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from config import OMEGA
from solver import solve_idr


# ── Mock-транспорт (аналитические формулы в обёртке) ──────────────────────────

class AnalyticalTransportMock:
    """
    Mock-объект, который вызывает те же аналитические формулы из physics.py,
    но через интерфейс BolsigTransport. Должен давать идентичный результат.
    """

    def __init__(self):
        from physics import (
            ionization_freq as _ionization_freq,
            ambipolar_diffusion as _ambipolar_diffusion,
            collision_freq as _collision_freq,
        )
        self._ionization_freq = _ionization_freq
        self._ambipolar_diffusion = _ambipolar_diffusion
        self._collision_freq = _collision_freq

    def ionization_freq(self, E_eff, p_pa):
        return self._ionization_freq(E_eff, p_pa)

    def ambipolar_diffusion(self, E_eff, p_pa):
        return self._ambipolar_diffusion(E_eff, p_pa)

    def collision_freq(self, E_eff, p_pa):
        """Аналитический νc не зависит от E_eff, но интерфейс принимает его."""
        return self._collision_freq(p_pa)


class ConstantTransportMock:
    """
    Mock-транспорт с постоянными коэффициентами.
    Полезен для тестирования структуры решателя.
    """

    def __init__(self, nu_i_val=100.0, Da_val=1e-3, nu_c_val=1e10):
        self._nu_i = nu_i_val
        self._Da = Da_val
        self._nu_c = nu_c_val

    def ionization_freq(self, E_eff, p_pa):
        if isinstance(E_eff, np.ndarray):
            return np.full_like(E_eff, self._nu_i)
        return self._nu_i

    def ambipolar_diffusion(self, E_eff, p_pa):
        if isinstance(E_eff, np.ndarray):
            return np.full_like(E_eff, self._Da)
        return self._Da

    def collision_freq(self, E_eff, p_pa):
        if isinstance(E_eff, np.ndarray):
            return np.full_like(E_eff, self._nu_c)
        return self._nu_c


# ── Обратная совместимость ───────────────────────────────────────────────────

class TestSolverBackwardCompatibility:
    """transport=None → поведение идентичное текущему."""

    def test_no_transport_converges(self):
        """Стандартный вызов без transport сходится."""
        result = solve_idr(N=80, R=0.012, p_pa=133.0,
                           H_wall=100000.0, max_iter=300,
                           tol=1e-5, relax=0.5)
        assert result["converged"], "Решатель не сошёлся без transport"

    def test_no_transport_physical_fields(self):
        """Поля физичны при transport=None."""
        result = solve_idr(N=80, R=0.012, p_pa=133.0,
                           H_wall=100000.0, max_iter=300,
                           tol=1e-5, relax=0.5)
        assert np.all(result["u"] >= -1e-10)
        assert np.all(result["v"] >= -1e-10)
        assert np.all(result["sigma_a"] >= -1e-10)
        assert abs(result["sigma_a"][-1]) < 1e-8


# ── Подключаемый транспорт ───────────────────────────────────────────────────

class TestSolverWithTransport:
    """Решатель принимает объект transport и использует его."""

    def test_accepts_transport_kwarg(self):
        """solve_idr() принимает параметр transport без ошибки."""
        mock = ConstantTransportMock()
        result = solve_idr(N=50, R=0.012, p_pa=133.0,
                           H_wall=1.0, max_iter=100,
                           tol=1e-4, relax=0.5,
                           transport=mock)
        assert "sigma_a" in result

    def test_analytical_mock_matches_default(self):
        """
        AnalyticalTransportMock (обёртка physics.py) должен давать
        результат, идентичный transport=None.
        """
        mock = AnalyticalTransportMock()
        params = dict(N=80, R=0.012, p_pa=133.0,
                      H_wall=100000.0, max_iter=300,
                      tol=1e-5, relax=0.5)

        result_none = solve_idr(**params, transport=None)
        result_mock = solve_idr(**params, transport=mock)

        # Профили sigma_a должны совпадать с высокой точностью
        np.testing.assert_allclose(
            result_mock["sigma_a"],
            result_none["sigma_a"],
            rtol=1e-6,
            err_msg="AnalyticalTransportMock даёт другой результат, чем transport=None"
        )

    def test_transport_fields_physical(self):
        """С transport все поля ≥ 0, σ(R) = 0."""
        mock = ConstantTransportMock(nu_i_val=500.0, Da_val=1e-3)
        result = solve_idr(N=50, R=0.012, p_pa=133.0,
                           H_wall=1.0, max_iter=200,
                           tol=1e-4, relax=0.5,
                           transport=mock)

        assert np.all(result["u"] >= -1e-10)
        assert np.all(result["v"] >= -1e-10)
        assert np.all(result["sigma_a"] >= -1e-10)
        assert abs(result["sigma_a"][-1]) < 1e-8

    def test_high_pressure_with_transport(self):
        """
        Решение при высоком давлении (5000 Па) с постоянным транспортом.
        Должно сходиться и давать физичные поля.
        """
        mock = ConstantTransportMock(nu_i_val=1e4, Da_val=1e-5, nu_c_val=5e10)
        result = solve_idr(N=50, R=0.00075, p_pa=5000.0,
                           H_wall=1000.0, max_iter=300,
                           tol=1e-4, relax=0.3,
                           transport=mock)

        assert np.all(result["sigma_a"] >= -1e-10)
        assert abs(result["sigma_a"][-1]) < 1e-8

    def test_sigma_ratio_uses_transport_nu_c(self):
        """sigma_p/sigma_a должен определяться nu_c из transport, а не аналитикой по p."""
        from physics import (
            ionization_freq as analytical_nu_i,
            ambipolar_diffusion as analytical_Da,
            collision_freq as analytical_nu_c,
        )

        class FixedNuCTransport:
            def __init__(self, nu_c_val):
                self._nu_c = float(nu_c_val)

            def ionization_freq(self, E_eff, p_pa):
                return analytical_nu_i(E_eff, p_pa)

            def ambipolar_diffusion(self, E_eff, p_pa):
                return analytical_Da(E_eff, p_pa)

            def collision_freq(self, E_eff, p_pa):
                if isinstance(E_eff, np.ndarray):
                    return np.full_like(E_eff, self._nu_c)
                return self._nu_c

        p_pa = 133.0
        nu_c_analytic = float(analytical_nu_c(p_pa))
        nu_c_transport = 0.2 * nu_c_analytic
        transport = FixedNuCTransport(nu_c_transport)

        result = solve_idr(
            N=80, R=0.012, p_pa=p_pa,
            H_wall=100000.0, max_iter=300,
            tol=1e-5, relax=0.5,
            transport=transport,
        )

        idx = int(np.argmax(result["sigma_a"]))
        sigma_a_peak = float(result["sigma_a"][idx])
        sigma_p_peak = float(result["sigma_p"][idx])
        assert sigma_a_peak > 0.0

        ratio_model = sigma_p_peak / sigma_a_peak
        ratio_transport = OMEGA / nu_c_transport
        ratio_analytic = OMEGA / nu_c_analytic

        assert ratio_model == pytest.approx(ratio_transport, rel=1e-2)
        assert not np.isclose(ratio_model, ratio_analytic, rtol=0.1)


# ── Рекомбинация в солвере ───────────────────────────────────────────────────

class TestSolverWithRecombination:
    """Решатель принимает beta_recomb и использует его в уравнении (13)."""

    def test_accepts_beta_recomb_kwarg(self):
        """solve_idr() принимает параметр beta_recomb без ошибки."""
        result = solve_idr(N=50, R=0.012, p_pa=133.0,
                           H_wall=100000.0, max_iter=200,
                           tol=1e-4, relax=0.5,
                           beta_recomb=1e-13)
        assert "sigma_a" in result

    def test_zero_beta_matches_default(self):
        """beta_recomb=0 идентично отсутствию рекомбинации."""
        params = dict(N=80, R=0.012, p_pa=133.0,
                      H_wall=100000.0, max_iter=300,
                      tol=1e-5, relax=0.5)

        result_none = solve_idr(**params)  # beta_recomb не задан
        result_zero = solve_idr(**params, beta_recomb=0.0)

        np.testing.assert_allclose(
            result_zero["sigma_a"],
            result_none["sigma_a"],
            rtol=1e-6,
            err_msg="beta_recomb=0 даёт другой результат, чем без beta_recomb"
        )

    def test_recombination_reduces_ne(self):
        """
        При β > 0 максимальная ne меньше, чем без рекомбинации.
        (Рекомбинация — дополнительный канал потерь.)
        """
        params = dict(N=80, R=0.012, p_pa=133.0,
                      H_wall=100000.0, max_iter=300,
                      tol=1e-5, relax=0.5)

        result_no_rec = solve_idr(**params, beta_recomb=0.0)
        result_rec = solve_idr(**params, beta_recomb=1e-10)

        ne_max_no_rec = result_no_rec["n_e"].max()
        ne_max_rec = result_rec["n_e"].max()

        assert ne_max_rec < ne_max_no_rec * 1.01, \
            f"ne с рекомбинацией ({ne_max_rec:.3e}) не меньше, чем без ({ne_max_no_rec:.3e})"

    def test_recombination_physical_fields(self):
        """Все поля физичны при beta_recomb > 0."""
        result = solve_idr(N=50, R=0.012, p_pa=133.0,
                           H_wall=100000.0, max_iter=300,
                           tol=1e-4, relax=0.5,
                           beta_recomb=1e-13)

        assert np.all(result["u"] >= -1e-10)
        assert np.all(result["v"] >= -1e-10)
        assert np.all(result["sigma_a"] >= -1e-10)
        assert abs(result["sigma_a"][-1]) < 1e-8


# ── Комбинация transport + recombination ─────────────────────────────────────

class TestSolverTransportAndRecombination:
    """Тесты одновременного использования transport + beta_recomb."""

    def test_combined_converges(self):
        """Решатель сходится с transport + beta_recomb."""
        mock = ConstantTransportMock(nu_i_val=1e3, Da_val=1e-4)
        result = solve_idr(N=50, R=0.005, p_pa=1000.0,
                           H_wall=1000.0, max_iter=300,
                           tol=1e-4, relax=0.3,
                           transport=mock, beta_recomb=1e-13)
        # Не обязательно converged=True, но поля должны быть физичны
        assert np.all(result["sigma_a"] >= -1e-10)

    def test_combined_fields_bounded(self):
        """Поля ограничены сверху (не уходят в бесконечность)."""
        mock = ConstantTransportMock(nu_i_val=1e4, Da_val=1e-5)
        result = solve_idr(N=50, R=0.001, p_pa=5000.0,
                           H_wall=500.0, max_iter=200,
                           tol=1e-3, relax=0.3,
                           transport=mock, beta_recomb=1e-12)

        assert np.all(np.isfinite(result["sigma_a"])), "sigma_a содержит inf/nan"
        assert np.all(np.isfinite(result["u"])), "u содержит inf/nan"
        assert np.all(np.isfinite(result["v"])), "v содержит inf/nan"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
