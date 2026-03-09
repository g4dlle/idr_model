"""
bolsig_parser.py — парсер выходных файлов BOLSIG+ (формат 4: таблица по E/N).

Формат 4 (E/N table) содержит таблицу с колонками:
  E/N (Td) | Mean energy (eV) | Mobility*N | Diffusion*N | Total collision freq./N |
  Total ionization freq./N | Total attachment freq./N | ...

Функция parse_bolsig_output(filepath) извлекает числовые данные и возвращает
словарь numpy-массивов.
"""

import re
import numpy as np


def parse_bolsig_output(filepath: str) -> dict[str, np.ndarray]:
    """
    Парсит выходной файл BOLSIG+ (Format 4) и извлекает транспортные
    и rate-коэффициенты как функции E/N.

    Parameters
    ----------
    filepath : путь к .dat-файлу (формат 4 — таблица по E/N)

    Returns
    -------
    dict с ключами:
        "E_N_Td"         : E/N в Таунсендах [Td]
        "mean_energy_eV" : средняя энергия электронов [эВ]
        "mobility_N"     : подвижность × N [1/(m·V·s)]
        "diffusion_N"    : коэфф. диффузии × N [1/(m·s)]
        "collision_N"    : частота столкновений / N [m³/s]
        "ionization_N"   : частота ионизации / N [m³/s]
        "attachment_N"   : частота прилипания / N [m³/s]  (если есть)

    Raises
    ------
    FileNotFoundError : файл не найден
    ValueError        : файл пуст или не содержит данных
    RuntimeError      : невозможно распарсить данные
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Файл пуст: {filepath}")

    # ── Найти начало таблицы данных ──────────────────────────────────────────
    # Таблица данных начинается после строки заголовка, содержащей
    # "Electric field / N (Td)" с числовыми данными ниже.
    # Ищем строку-заголовок таблицы и читаем числа после неё.

    data_lines = []
    header_found = False
    header_columns = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Ищем строку заголовка с "Electric field" или "E/N"
        if not header_found:
            if ("Electric field / N" in stripped or "E/N" in stripped) and \
               ("Mobility" in stripped or "Mean energy" in stripped):
                header_found = True
                # Парсим имена колонок (разделитель — табуляция)
                header_columns = [c.strip() for c in stripped.split("\t") if c.strip()]
                continue
        else:
            # После заголовка читаем строки с числами
            if not stripped:
                # Пустая строка после данных — конец таблицы
                if data_lines:
                    break
                continue

            # Проверяем, является ли строка числовой
            parts = stripped.split()
            try:
                float(parts[0])
                data_lines.append(stripped)
            except (ValueError, IndexError):
                # Не числовая строка — пропускаем или конец таблицы
                if data_lines:
                    break

    if not data_lines:
        raise RuntimeError(
            f"Не удалось найти числовые данные в файле: {filepath}. "
            f"Ожидается формат 4 (E/N table) BOLSIG+."
        )

    # ── Парсим числовые данные ───────────────────────────────────────────────
    rows = []
    for line in data_lines:
        parts = line.split()
        try:
            row = [float(x) for x in parts]
            rows.append(row)
        except ValueError:
            continue  # пропускаем нечисловые строки

    if not rows:
        raise RuntimeError(f"Не удалось извлечь числовые данные из файла: {filepath}")

    data = np.array(rows)
    n_cols = data.shape[1]

    # ── Маппинг колонок ──────────────────────────────────────────────────────
    # Минимум 6 колонок: E/N, mean energy, mobility*N, diffusion*N,
    #                     collision freq/N, ionization freq/N
    result = {}
    result["E_N_Td"] = data[:, 0]
    result["mean_energy_eV"] = data[:, 1]

    if n_cols >= 3:
        result["mobility_N"] = data[:, 2]
    if n_cols >= 4:
        result["diffusion_N"] = data[:, 3]
    if n_cols >= 5:
        result["collision_N"] = data[:, 4]
    if n_cols >= 6:
        result["ionization_N"] = data[:, 5]
    if n_cols >= 7:
        result["attachment_N"] = data[:, 6]

    # Валидация минимальных требований
    required = ["E_N_Td", "mean_energy_eV", "mobility_N", "diffusion_N", "ionization_N"]
    for key in required:
        if key not in result:
            raise RuntimeError(
                f"Недостаточно колонок в данных ({n_cols}). "
                f"Ключ '{key}' не удалось извлечь. "
                f"Ожидается минимум 6 колонок в формате 4 BOLSIG+."
            )

    return result
