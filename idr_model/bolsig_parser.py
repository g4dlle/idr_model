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
    Парсит выходной файл BOLSIG+ (Format 4) и извлекает:
    - E/N [Td]
    - Mean energy [eV]
    - Mobility*N [1/m/V/s]
    - Diffusion*N [1/m/s]
    - Momentum frequency /N [m3/s]
    - Total ionization freq. /N [m3/s]
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Файл пуст: {filepath}")

    blocks = {}
    current_block = None
    en_array = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_block = None
            continue
            
        if stripped.startswith("E/N (Td)"):
            parts = stripped.split("\t")
            if len(parts) >= 2:
                current_block = parts[1].strip()
                if current_block not in blocks:
                    blocks[current_block] = ([], [])
            continue
            
        if current_block:
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    blocks[current_block][0].append(x)
                    blocks[current_block][1].append(y)
                except ValueError:
                    current_block = None

    if not blocks:
        raise RuntimeError(f"Не удалось найти числовые данные в файле: {filepath}")

    result = {}
    
    # Ищем нужные блоки
    key_mapping = {
        "mean_energy_eV": ["Mean energy (eV)"],
        "mobility_N": ["Mobility *N (1/m/V/s)", "Re/perp mobility *N (1/m/V/s)"],
        "diffusion_N": ["Diffusion coefficient *N (1/m/s)"],
        "collision_N": ["Momentum frequency /N (m3/s)", "Total collision freq. /N (m3/s)", "Effective (momentum) freq. /N (m3/s)"],
        "ionization_N": ["Total ionization freq. /N (m3/s)"]
    }
    
    # E/N берем из любого блока
    first_block = list(blocks.values())[0]
    result["E_N_Td"] = np.array(first_block[0])
    
    for key, possible_names in key_mapping.items():
        found = False
        for name in possible_names:
            if name in blocks:
                result[key] = np.array(blocks[name][1])
                found = True
                break
        if not found and key != "attachment_N":
             pass # Will be caught by validation

    # Валидация
    required = ["E_N_Td", "mean_energy_eV", "mobility_N", "diffusion_N", "ionization_N"]
    for key in required:
        if key not in result:
             raise RuntimeError(f"Не найден обязательный параметр: {key} (возможно не тот формат BOLSIG+)")

    return result
