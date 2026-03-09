"""
test_bolsig_parser.py — тесты парсера выходных файлов BOLSIG+.

Тестируемый модуль: bolsig_parser.py
Формат входа: Format 4 (таблица по E/N) из bolsigminus.exe.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_FILE = os.path.join(FIXTURE_DIR, "bolsig_output_ar.dat")


@pytest.fixture
def parsed_data():
    """Парсит mock-файл BOLSIG+ и возвращает результат."""
    from bolsig_parser import parse_bolsig_output
    return parse_bolsig_output(FIXTURE_FILE)


# ── Базовые тесты структуры ──────────────────────────────────────────────────

class TestParserStructure:
    """Проверка, что парсер извлекает правильную структуру данных."""

    def test_returns_dict(self, parsed_data):
        assert isinstance(parsed_data, dict)

    def test_contains_required_keys(self, parsed_data):
        """Все необходимые для модели ключи присутствуют."""
        required_keys = [
            "E_N_Td",           # E/N в Таунсендах
            "mean_energy_eV",   # средняя энергия электронов
            "mobility_N",       # μ×N [1/(m·V·s)]
            "diffusion_N",      # D×N [1/(m·s)]
            "ionization_N",     # kion×N = νi/N [m³/s]  (!! чтот так обзначают)
        ]
        for key in required_keys:
            assert key in parsed_data, f"Ключ '{key}' отсутствует в результате парсера"

    def test_all_values_are_arrays(self, parsed_data):
        for key, val in parsed_data.items():
            assert isinstance(val, np.ndarray), f"Ключ '{key}': ожидался np.ndarray, получен {type(val)}"

    def test_all_arrays_same_length(self, parsed_data):
        lengths = {key: len(val) for key, val in parsed_data.items()}
        unique_lengths = set(lengths.values())
        assert len(unique_lengths) == 1, f"Массивы имеют разную длину: {lengths}"

    def test_correct_number_of_points(self, parsed_data):
        """Mock-файл содержит 13 точек E/N."""
        assert len(parsed_data["E_N_Td"]) == 13


# ── Физические ограничения на значения ───────────────────────────────────────

class TestParserPhysicalValues:
    """Проверка физической корректности извлечённых данных."""

    def test_extracted_arrays_shape(self, parsed_data):
        assert len(parsed_data["E_N_Td"]) == 50, "Ожидается 50 точек E/N"

    def test_E_N_positive_and_sorted(self, parsed_data):
        E_N = parsed_data["E_N_Td"]
        assert np.all(E_N > 0), f"E/N содержит неположительные значения: {E_N.min()}"
        assert np.all(np.diff(E_N) > 0), "E/N не отсортирован по возрастанию"

    def test_E_N_range(self, parsed_data):
        """E/N должно быть в диапазоне 1–1000 Td (из mock-файла)."""
        E_N = parsed_data["E_N_Td"]
        assert E_N.min() == pytest.approx(0.1, rel=1e-3), "Неверная минимальная граница E/N"
        assert E_N[-1] == pytest.approx(1000.0, rel=1e-3)

    def test_mean_energy_positive(self, parsed_data):
        assert np.all(parsed_data["mean_energy_eV"] > 0)

    def test_mean_energy_increases_with_E_N(self, parsed_data):
        """Средняя энергия должна расти с E/N (больше поле → горячее электроны)."""
        energies = parsed_data["mean_energy_eV"]
        max_e = np.max(energies)
        assert max_e < 20.0, "Нереалистично высокая средняя энергия (>20 eV)"
        assert np.all(np.diff(energies) > 0), "Средняя энергия не растёт с E/N"

    def test_mobility_N_positive(self, parsed_data):
        """Подвижность должна быть положительной."""
        mob = parsed_data["mobility_N"]
        assert np.all(mob > 0), "Подвижность неположительна"

    def test_diffusion_N_positive(self, parsed_data):
        """Диффузия должна быть положительной."""
        diff = parsed_data["diffusion_N"]
        assert np.all(diff > 0), "Диффузия неположительна"

    def test_ionization_nonnegative(self, parsed_data):
        """Частота ионизации ≥ 0 (может быть = 0 при малых E/N)."""
        assert np.all(parsed_data["ionization_N"] >= 0)

    def test_ionization_zero_at_low_E_N(self, parsed_data):
        """При E/N < 50 Td для аргона ионизация практически нулевая."""
        E_N = parsed_data["E_N_Td"]
        # Ионизация должна быть (почти) нулевой при очень низких E/N (< 1 Td)
        low_en_mask = E_N < 1.0
        assert np.all(parsed_data["ionization_N"][low_en_mask] < 1e-30), \
            f"Ионизация не нулевая при низких E/N: {parsed_data['ionization_N'][low_en_mask]}"

    def test_ionization_grows_with_E_N(self, parsed_data):
        """При E/N > порога ионизации, kion растёт с E/N."""
        E_N = parsed_data["E_N_Td"]
        kion = parsed_data["ionization_N"]
        # Берём только точки, где ионизация > 0
        nonzero_mask = kion > 0
        if np.sum(nonzero_mask) >= 2:
            kion_nz = kion[nonzero_mask]
            assert np.all(np.diff(kion_nz) > 0), \
                "kion не растёт с E/N в области ионизации"


# ── Обработка ошибок ─────────────────────────────────────────────────────────

class TestParserErrors:

    def test_nonexistent_file_raises(self):
        from bolsig_parser import parse_bolsig_output
        with pytest.raises((FileNotFoundError, OSError)):
            parse_bolsig_output("/nonexistent/path/file.dat")

    def test_empty_file_raises(self, tmp_path):
        from bolsig_parser import parse_bolsig_output
        empty = tmp_path / "empty.dat"
        empty.write_text("")
        with pytest.raises((ValueError, RuntimeError)):
            parse_bolsig_output(str(empty))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
