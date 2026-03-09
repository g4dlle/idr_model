"""
test_bolsig_transport.py — тесты интерполятора транспортных коэффициентов BOLSIG+.

Тестируемый модуль: bolsig_transport.py
Класс: BolsigTransport

Интерфейс BolsigTransport должен предоставлять методы:
  - ionization_freq(E_eff, p_pa) → νi [с⁻¹]
  - ambipolar_diffusion(E_eff, p_pa) → Da [м²/с]
  - collision_freq(E_eff, p_pa) → νc [с⁻¹]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

# Физические константы
K_BOLTZMANN = 1.380649e-23     # Дж/К
T_GAS = 300.0                  # К
P_PA_TEST = 1000.0             # Па (давление для тестов)
N_GAS_TEST = P_PA_TEST / (K_BOLTZMANN * T_GAS)  # м⁻³ (≈ 2.41×10²³)


@pytest.fixture
def mock_bolsig_data():
    """Создаёт mock-данные, имитирующие результат parse_bolsig_output()."""
    E_N = np.array([1, 3, 10, 30, 100, 300, 1000], dtype=float)  # Td
    return {
        "E_N_Td":         E_N,
        "mean_energy_eV": np.array([0.04, 0.07, 0.2, 0.7, 2.5, 9.1, 30.0]),
        "mobility_N":     np.array([4.2e24, 2.5e24, 1.3e24, 5.9e23, 2.7e23, 1.2e23, 5.5e22]),
        "diffusion_N":    np.array([2.5e23, 3.0e23, 4.0e23, 6.3e23, 9.9e23, 1.5e24, 2.0e24]),
        "ionization_N":   np.array([0.0,    0.0,    0.0,    0.0,    5.4e-16, 2.3e-13, 6.8e-12]),
    }


@pytest.fixture
def transport(mock_bolsig_data):
    """Создаёт объект BolsigTransport из mock-данных."""
    from bolsig_transport import BolsigTransport
    return BolsigTransport(mock_bolsig_data, p_pa=P_PA_TEST, T_gas=T_GAS)


# ── Тесты инициализации ─────────────────────────────────────────────────────

class TestTransportInit:

    def test_creates_without_error(self, mock_bolsig_data):
        from bolsig_transport import BolsigTransport
        t = BolsigTransport(mock_bolsig_data, p_pa=P_PA_TEST)
        assert t is not None

    def test_stores_pressure(self, transport):
        assert transport.p_pa == P_PA_TEST

    def test_rejects_empty_data(self):
        from bolsig_transport import BolsigTransport
        with pytest.raises((ValueError, KeyError)):
            BolsigTransport({}, p_pa=P_PA_TEST)


# ── Тесты ionization_freq ───────────────────────────────────────────────────

class TestIonizationFreq:

    def test_returns_array(self, transport):
        E_eff = np.array([100.0, 500.0, 1000.0])
        result = transport.ionization_freq(E_eff, P_PA_TEST)
        assert isinstance(result, np.ndarray)
        assert result.shape == E_eff.shape

    def test_returns_scalar_for_scalar(self, transport):
        result = transport.ionization_freq(1000.0, P_PA_TEST)
        assert np.isscalar(result) or result.ndim == 0

    def test_nonnegative(self, transport):
        E_eff = np.linspace(0.0, 10000.0, 100)
        result = transport.ionization_freq(E_eff, P_PA_TEST)
        assert np.all(result >= 0), f"νi содержит отрицательные: {result.min()}"

    def test_zero_at_zero_field(self, transport):
        """При E_eff = 0 ионизации нет."""
        result = transport.ionization_freq(0.0, P_PA_TEST)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_small_at_low_field(self, transport):
        """При малых E_eff ионизация пренебрежимо мала."""
        E_eff = 10.0  # В/м → E/N ~ 4e-23 Td — далеко ниже порога
        result = transport.ionization_freq(E_eff, P_PA_TEST)
        assert result < 1.0, f"νi слишком велика при E_eff={E_eff}: {result}"

    def test_grows_with_field(self, transport):
        """νi монотонно растёт с E_eff (в области ионизации)."""
        E_eff = np.array([1000.0, 5000.0, 10000.0, 50000.0])
        result = transport.ionization_freq(E_eff, P_PA_TEST)
        nonzero = result > 0
        if np.sum(nonzero) >= 2:
            assert np.all(np.diff(result[nonzero]) > 0), \
                f"νi не растёт с E_eff: {result}"


# ── Тесты ambipolar_diffusion ───────────────────────────────────────────────

class TestAmbipolarDiffusion:

    def test_returns_array(self, transport):
        E_eff = np.array([100.0, 500.0])
        result = transport.ambipolar_diffusion(E_eff, P_PA_TEST)
        assert isinstance(result, np.ndarray)

    def test_positive(self, transport):
        E_eff = np.linspace(0.0, 10000.0, 50)
        result = transport.ambipolar_diffusion(E_eff, P_PA_TEST)
        assert np.all(result > 0), f"Da содержит неположительные: {result.min()}"

    def test_inversely_proportional_to_pressure(self, mock_bolsig_data):
        """Da ∝ 1/p при фиксированном E/N (т.к. D×N = const по E/N)."""
        from bolsig_transport import BolsigTransport

        p1, p2 = 1000.0, 2000.0
        t1 = BolsigTransport(mock_bolsig_data, p_pa=p1, T_gas=T_GAS)
        t2 = BolsigTransport(mock_bolsig_data, p_pa=p2, T_gas=T_GAS)

        # E_eff подбираем так, чтобы E/N попадало в одну и ту же точку
        # E/N = E_eff / n_gas; n_gas ∝ p → E_eff ∝ p для того же E/N
        E_N_target = 100.0  # Td = 100×1e-21 V·m²
        n_gas_1 = p1 / (K_BOLTZMANN * T_GAS)
        n_gas_2 = p2 / (K_BOLTZMANN * T_GAS)
        E_eff_1 = E_N_target * 1e-21 * n_gas_1  # В/м
        E_eff_2 = E_N_target * 1e-21 * n_gas_2

        Da1 = t1.ambipolar_diffusion(E_eff_1, p1)
        Da2 = t2.ambipolar_diffusion(E_eff_2, p2)

        # Da1/Da2 ≈ p2/p1 = 2
        ratio = float(Da1) / float(Da2)
        assert ratio == pytest.approx(2.0, rel=0.1), \
            f"Da не обратно пропорциональна давлению: Da1/Da2 = {ratio:.3f}, ожидалось 2.0"


# ── Тесты collision_freq ────────────────────────────────────────────────────

class TestCollisionFreq:

    def test_positive(self, transport):
        E_eff = np.array([0.0, 100.0, 1000.0])
        result = transport.collision_freq(E_eff, P_PA_TEST)
        assert np.all(result > 0), f"νc содержит неположительные: {result.min()}"

    def test_reasonable_order_of_magnitude(self, transport):
        """
        Для аргона при 1000 Па (≈7.5 торр, T=300K):
        νc ≈ 5.6e9 × 7.5 ≈ 4.2×10¹⁰ с⁻¹  (оценка из physics.py)
        BOLSIG+ должен давать порядок 10⁹–10¹¹.
        """
        result = transport.collision_freq(1000.0, P_PA_TEST)
        assert 1e8 < float(result) < 1e12, \
            f"νc = {float(result):.2e} вне ожидаемого диапазона 10⁸–10¹²"

    def test_proportional_to_pressure(self, mock_bolsig_data):
        """νc ∝ n_gas ∝ p при фиксированном E/N."""
        from bolsig_transport import BolsigTransport

        p1, p2 = 1000.0, 5000.0
        t1 = BolsigTransport(mock_bolsig_data, p_pa=p1, T_gas=T_GAS)
        t2 = BolsigTransport(mock_bolsig_data, p_pa=p2, T_gas=T_GAS)

        # Одинаковый E/N → одинаковый E_eff/n_gas
        E_N_target = 50.0  # Td
        n1 = p1 / (K_BOLTZMANN * T_GAS)
        n2 = p2 / (K_BOLTZMANN * T_GAS)
        E1 = E_N_target * 1e-21 * n1
        E2 = E_N_target * 1e-21 * n2

        nc1 = float(t1.collision_freq(E1, p1))
        nc2 = float(t2.collision_freq(E2, p2))

        ratio = nc2 / nc1
        assert ratio == pytest.approx(5.0, rel=0.1), \
            f"νc не пропорциональна давлению: nc2/nc1 = {ratio:.3f}, ожидалось 5.0"


# ── Тесты гладкости интерполяции ─────────────────────────────────────────────

class TestInterpolationSmoothness:

    def test_ionization_smooth(self, transport):
        """Интерполяция νi не содержит скачков (проверка на мелкой сетке)."""
        E_eff = np.linspace(1000.0, 50000.0, 200)
        nu_i = transport.ionization_freq(E_eff, P_PA_TEST)
        # Относительные разности между соседними точками < 50%
        nonzero = nu_i > 1e-10
        if np.sum(nonzero) > 2:
            nu_nz = nu_i[nonzero]
            rel_jumps = np.abs(np.diff(nu_nz)) / (nu_nz[:-1] + 1e-30)
            max_jump = rel_jumps.max()
            assert max_jump < 0.5, \
                f"Скачок в интерполяции νi: макс. отн. разность = {max_jump:.3f}"

    def test_diffusion_smooth(self, transport):
        """Интерполяция Da не содержит скачков."""
        E_eff = np.linspace(100.0, 50000.0, 200)
        Da = transport.ambipolar_diffusion(E_eff, P_PA_TEST)
        rel_jumps = np.abs(np.diff(Da)) / (Da[:-1] + 1e-30)
        max_jump = rel_jumps.max()
        assert max_jump < 0.5, \
            f"Скачок в интерполяции Da: макс. отн. разность = {max_jump:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
