"""Utilities for reading BOLSIG+ swarm tables.

The project uses a BOLSIG+ ``SAVERESULTS`` file in format 4 (E/N tables).
Only the fields needed by the plasma model are parsed:

* mean electron energy;
* mobility times neutral density, ``mu*N``;
* diffusion coefficient times neutral density, ``D*N``;
* total ionization frequency divided by neutral density, ``nu_i/N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BolsigTable:
    e_over_n_td: np.ndarray
    mean_energy_ev: np.ndarray
    mobility_n: np.ndarray
    diffusion_n: np.ndarray
    ionization_freq_over_n: np.ndarray

    def interp(self, values: np.ndarray, e_over_n_td):
        """Log-log interpolation for positive BOLSIG coefficients."""
        x = np.asarray(e_over_n_td, dtype=float)
        x_safe = np.maximum(x, self.e_over_n_td[0])
        x_safe = np.minimum(x_safe, self.e_over_n_td[-1])

        y = np.asarray(values, dtype=float)
        if np.all(y > 0):
            out = np.exp(np.interp(np.log(x_safe),
                                   np.log(self.e_over_n_td),
                                   np.log(y)))
        else:
            out = np.interp(x_safe, self.e_over_n_td, y)
        return out


def _read_numeric_block(lines: list[str], start: int) -> tuple[np.ndarray, np.ndarray]:
    x_vals = []
    y_vals = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            break
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            break
        x_vals.append(x)
        y_vals.append(y)
    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def _find_block(lines: list[str], title_part: str) -> tuple[np.ndarray, np.ndarray]:
    for i, line in enumerate(lines):
        if line.startswith("E/N (Td)") and title_part in line:
            x, y = _read_numeric_block(lines, i)
            if len(x) > 0:
                return x, y
    raise ValueError(f"BOLSIG block not found: {title_part}")


@lru_cache(maxsize=4)
def load_bolsig_table(path: str) -> BolsigTable:
    """Load a BOLSIG+ format-4 result file."""
    table_path = Path(path)
    if not table_path.is_absolute():
        table_path = Path(__file__).resolve().parent / table_path
    table_path = table_path.resolve()

    lines = table_path.read_text(encoding="utf-8", errors="replace").splitlines()

    e_mean, mean_energy = _find_block(lines, "Mean energy")
    e_mob, mobility_n = _find_block(lines, "Mobility *N")
    e_diff, diffusion_n = _find_block(lines, "Diffusion coefficient *N")
    e_ion, ion_freq_over_n = _find_block(lines, "Total ionization freq. /N")

    if not (np.allclose(e_mean, e_mob)
            and np.allclose(e_mean, e_diff)
            and np.allclose(e_mean, e_ion)):
        raise ValueError("BOLSIG table blocks use different E/N grids")

    return BolsigTable(
        e_over_n_td=e_mean,
        mean_energy_ev=mean_energy,
        mobility_n=mobility_n,
        diffusion_n=diffusion_n,
        ionization_freq_over_n=ion_freq_over_n,
    )
