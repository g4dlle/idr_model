"""Проверка H(r) из реальной модели и сравнение с экспериментом Рис. 2.27."""
import sys; sys.path.insert(0, 'idr_model')
import numpy as np
from idr_model.self_consistent import solve_maxwell_for_ne0
from idr_model.physics import conductivity
from idr_model import config as cfg

mu0   = 4 * np.pi * 1e-7
omega = cfg.OMEGA
R     = 0.012
p_pa  = 133.0

# Экспериментальные точки (Рис. 2.27, кривая 3, аргон G=0)
r_exp_mm = [0.19, 4.74, 8.64, 11.81]
H_exp    = [41.50, 41.64, 43.93, 54.59]   # ×10² А/м
H_exp_norm = [h / H_exp[-1] for h in H_exp]

def skin_depth(ne, p):
    sa, _, _ = conductivity(ne, p)
    return (2 / (omega * mu0 * sa))**0.5 if sa > 0 else 999.0

print("=" * 65)
print("ПРОФИЛЬ H(r)/H(R): РЕАЛЬНАЯ МОДЕЛЬ vs ЭКСПЕРИМЕНТ (Рис. 2.27)")
print("=" * 65)
print()

test_cases = [
    ("Эксп. n_e (Рис.2.38, 1.5 кВт)", 3.71e17),
    ("Эксп. n_e (Рис.2.38, 2.5 кВт)", 7.83e17),
    ("Эксп. n_e (Рис.2.38, 3.2 кВт)", 22.4e17),
    ("Модель n_e0* (Table1)",          3.875e20),
]

r_query = np.array([0.0, 4e-3, 8e-3, 12e-3])

for label, ne in test_cases:
    res = solve_maxwell_for_ne0(n_e0=ne, R=R, p_pa=p_pa, H_wall=cfg.H_WALL)
    r_grid = res['r']
    H_abs  = np.sqrt(np.maximum(res['u'], 0.0))
    H_norm = H_abs / H_abs[-1] if H_abs[-1] > 0 else H_abs

    H_at_pts = np.interp(r_query, r_grid, H_norm)
    delta = skin_depth(ne, p_pa)
    lam0  = res['lambda0']

    print(f"  {label}")
    print(f"    n_e0 = {ne:.3e} m^-3 | delta/R = {delta/R:.2f} | lam0 = {lam0:.4f}")
    print(f"    r, mm  | H_model | H_exp")
    for i, rm in enumerate([0, 4, 8, 12]):
        print(f"    {rm:5d}  | {H_at_pts[i]:.4f}  | {H_exp_norm[i]:.4f}")
    print()

print("-" * 65)
print("ВЫВОД:")
print("  Правая (стенка) нормировка H(R)=1 одинакова для всех.")
print("  Отличие в центре (r=0) = скин-эффект.")
