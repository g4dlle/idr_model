"""
bolsig_transport.py — интерполятор транспортных коэффициентов из данных BOLSIG+.

Класс BolsigTransport предоставляет методы с сигнатурой, совпадающей
с функциями из physics.py, чтобы можно было «подставить» объект
в solve_idr() через параметр transport=.

Внутренне использует log-log интерполяцию по E/N для гладкости
на нескольких порядках величины.
"""

import numpy as np
from scipy.interpolate import interp1d

# Физические константы
K_BOLTZMANN = 1.380649e-23    # Дж/К
E_CHARGE = 1.602176634e-19    # Кл
M_ELECTRON = 9.1093837015e-31 # кг
TD_TO_VM2 = 1e-21             # 1 Td = 1e-21 В·м²


class BolsigTransport:
    """
    Интерполятор транспортных коэффициентов на основе данных BOLSIG+.

    Методы:
      ionization_freq(E_eff, p_pa) → νi [с⁻¹]
      ambipolar_diffusion(E_eff, p_pa) → Da [м²/с]
      collision_freq(E_eff, p_pa) → νc [с⁻¹]

    Parameters
    ----------
    bolsig_data : dict из parse_bolsig_output()
        Должен содержать: "E_N_Td", "mobility_N", "diffusion_N", "ionization_N"
    p_pa : float
        Давление [Па] (определяет n_gas для пересчёта E_eff → E/N)
    T_gas : float
        Температура газа [К] (по умолчанию 300)
    """

    def __init__(self, bolsig_data: dict, p_pa: float, T_gas: float = 300.0):
        # Валидация
        required = ["E_N_Td", "mobility_N", "diffusion_N", "ionization_N"]
        for key in required:
            if key not in bolsig_data:
                raise ValueError(f"Ключ '{key}' отсутствует в bolsig_data")

        if len(bolsig_data["E_N_Td"]) < 2:
            raise ValueError("Нужно минимум 2 точки для интерполяции")

        self.p_pa = p_pa
        self.T_gas = T_gas
        self.n_gas = p_pa / (K_BOLTZMANN * T_gas)  # м⁻³

        E_N = bolsig_data["E_N_Td"]
        self._E_N_min = E_N.min()
        self._E_N_max = E_N.max()

        # ── Log-log интерполяторы для гладких функций ────────────────────────
        log_EN = np.log(E_N)

        # Подвижность × N → частота столкновений / N
        # νc = e / (me × μ), где μ = mobility_N / N
        # Но BOLSIG+ даёт mobility*N напрямую, и collision_N если есть
        mu_N = bolsig_data["mobility_N"]  # [1/(m·V·s)]
        self._interp_log_mu_N = interp1d(
            log_EN, np.log(mu_N),
            kind="linear", fill_value="extrapolate"
        )

        # Коэфф. диффузии × N
        D_N = bolsig_data["diffusion_N"]  # [1/(m·s)]
        self._interp_log_D_N = interp1d(
            log_EN, np.log(D_N),
            kind="linear", fill_value="extrapolate"
        )

        # Частота столкновений / N (если есть в данных)
        if "collision_N" in bolsig_data and np.any(bolsig_data["collision_N"] > 0):
            coll_N = bolsig_data["collision_N"]
            self._interp_log_coll_N = interp1d(
                log_EN, np.log(np.maximum(coll_N, 1e-30)),
                kind="linear", fill_value="extrapolate"
            )
            self._has_collision = True
        else:
            self._has_collision = False

        # Частота ионизации / N
        # Особенность: kion = 0 при малых E/N. Используем линейную
        # интерполяцию в log-пространстве только для ненулевых точек.
        kion_N = bolsig_data["ionization_N"]  # [m³/s]
        nonzero = kion_N > 0
        if np.any(nonzero):
            E_N_nz = E_N[nonzero]
            kion_nz = kion_N[nonzero]
            self._E_N_ion_threshold = E_N_nz[0]
            self._interp_log_kion_N = interp1d(
                np.log(E_N_nz), np.log(kion_nz),
                kind="linear", fill_value="extrapolate"
            )
            self._has_ionization = True
        else:
            self._has_ionization = False
            self._E_N_ion_threshold = np.inf

    def _E_eff_to_E_N(self, E_eff, p_pa):
        """Пересчёт E_eff [В/м] → E/N [Td] при текущем p_pa."""
        n_gas = p_pa / (K_BOLTZMANN * self.T_gas)
        return E_eff / (n_gas * TD_TO_VM2)

    def ionization_freq(self, E_eff, p_pa):
        """
        Частота ионизации νi [с⁻¹].

        νi = kion(E/N) × n_gas, где kion из BOLSIG+.
        """
        E_eff = np.asarray(E_eff, dtype=float)
        scalar = E_eff.ndim == 0
        E_eff = np.atleast_1d(E_eff)

        n_gas = p_pa / (K_BOLTZMANN * self.T_gas)
        E_N = self._E_eff_to_E_N(E_eff, p_pa)

        result = np.zeros_like(E_eff)

        if self._has_ionization:
            above = E_N >= self._E_N_ion_threshold
            if np.any(above):
                log_kion = self._interp_log_kion_N(np.log(E_N[above]))
                kion = np.exp(log_kion)
                result[above] = kion * n_gas

        # E_eff = 0 → νi = 0
        result[E_eff <= 0] = 0.0

        return float(result[0]) if scalar else result

    def ambipolar_diffusion(self, E_eff, p_pa):
        """
        Амбиполярный коэффициент диффузии Da [м²/с].

        Da ≈ D_e × (1 + Te/Ti) ≈ 2 × D_e  (Te >> Ti обычно, но для простоты ×2)
        D_e = D_N / N, где D_N из BOLSIG+.
        Da = 2 × D_N / n_gas  (грубая оценка, Te/Ti ≈ 1 при низких энергиях)
        """
        E_eff = np.asarray(E_eff, dtype=float)
        scalar = E_eff.ndim == 0
        E_eff = np.atleast_1d(E_eff)

        n_gas = p_pa / (K_BOLTZMANN * self.T_gas)
        E_N = self._E_eff_to_E_N(E_eff, p_pa)

        # Clamp E/N to table range for extrapolation safety
        log_EN = np.log(np.maximum(E_N, self._E_N_min * 0.1))
        log_DN = self._interp_log_D_N(log_EN)
        D_N = np.exp(log_DN)

        # Da = D_N / n_gas (электронная диффузия ≈ амбиполярная при Te ≈ Ti)
        # Для более точной оценки нужно Te/Ti, но на первом этапе Da ≈ D_N/n_gas
        Da = D_N / n_gas

        return float(Da[0]) if scalar else Da

    def collision_freq(self, E_eff, p_pa):
        """
        Частота упругих столкновений νc [с⁻¹].

        Если BOLSIG+ предоставил collision_N:
            νc = collision_N × n_gas
        Иначе вычисляем из подвижности:
            νc = e / (me × μ) = e × n_gas / (me × mobility_N)
        """
        E_eff = np.asarray(E_eff, dtype=float)
        scalar = E_eff.ndim == 0
        E_eff = np.atleast_1d(E_eff)

        n_gas = p_pa / (K_BOLTZMANN * self.T_gas)
        E_N = self._E_eff_to_E_N(E_eff, p_pa)

        log_EN = np.log(np.maximum(E_N, self._E_N_min * 0.1))

        if self._has_collision:
            log_cN = self._interp_log_coll_N(log_EN)
            nu_c = np.exp(log_cN) * n_gas
        else:
            # Из подвижности: μ = mobility_N / N → νc = e/(me·μ)
            log_muN = self._interp_log_mu_N(log_EN)
            mu_N = np.exp(log_muN)
            mu = mu_N / n_gas  # [m²/(V·s)]
            nu_c = E_CHARGE / (M_ELECTRON * mu)

        return float(nu_c[0]) if scalar else nu_c
