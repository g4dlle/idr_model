"""
Комплексная проверка кода и сравнения с экспериментом.
Запуск: uv run python check_code.py
"""
import sys; sys.path.insert(0, 'idr_model')
import numpy as np
from idr_model.self_consistent import solve_maxwell_for_ne0, compute_lambda0
from idr_model.equations import make_grid
from idr_model.physics import conductivity, effective_field, ionization_freq, ambipolar_diffusion
from idr_model import config as cfg

mu0   = 4 * np.pi * 1e-7
omega = cfg.OMEGA
R     = 0.012
p_pa  = 133.0

# ─────────────────────────────────────────────────────────
# ПРОВЕРКА 1: lambda0 vs n_e0 — правильная монотонность?
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. lambda0 vs n_e0 (p=133 Pa, R=12 mm, H_wall=1e5 A/m)")
print("   Ожидание: lambda0=1 при n_e0=n_e0*")
print("=" * 60)
ne_scan = [1e14, 1e16, 1e17, 3.71e17, 1e18, 1e19, 1e20, 3.875e20, 1e21]
print(f"{'n_e0':>12}  {'lambda0':>9}  {'conv':>5}  {'delta/R':>8}  {'nu_i/loss_wall':>16}")
for ne in ne_scan:
    res  = solve_maxwell_for_ne0(n_e0=ne, R=R, p_pa=p_pa, H_wall=cfg.H_WALL)
    sa, _, _ = conductivity(ne, p_pa)
    delta = (2/(omega*mu0*sa))**0.5
    # nu_i/loss at wall (r=R, E_eff=E_wall)
    E_wall = omega * mu0 * cfg.H_WALL * R / 2
    E_eff  = effective_field(E_wall, p_pa)
    ni     = ionization_freq(E_eff, p_pa)
    Da     = ambipolar_diffusion(E_eff, p_pa)
    ratio  = ni / (Da * 2.405**2 / R**2)
    print(f"{ne:12.3e}  {res['lambda0']:9.4f}  {str(res['converged']):>5}  {delta/R:8.2f}  {ratio:16.2f}")

# ─────────────────────────────────────────────────────────
# ПРОВЕРКА 2: lambda0 = mu или 1/mu?
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. ПРОВЕРКА: lambda0_sq = mu или 1/mu в compute_lambda0?")
print("   Тест: равномерные Da и nu_i (аналитическое решение)")
print("   Аналитически: lambda0^2 = nu_i*R^2 / (Da*2.405^2)")
print("=" * 60)
# Используем умеренное поле — нужен баланс
E_test = 3000.0  # В/м
p_test = 133.0
Da_test  = ambipolar_diffusion(effective_field(E_test, p_test), p_test)
ni_test  = ionization_freq(effective_field(E_test, p_test), p_test)
lambda_analytic_sq = ni_test * R**2 / (Da_test * 2.405**2)
print(f"  Da={Da_test:.4e} m2/s,  nu_i={ni_test:.4e} s^-1")
print(f"  lambda0^2 analytic (=mu) = {lambda_analytic_sq:.6f}")
print(f"  lambda0   analytic (=sqrt(mu)) = {lambda_analytic_sq**0.5:.6f}")

# Запустить compute_lambda0 с равномерными коэффициентами
r_grid, h = make_grid(100, R)
Da_arr  = np.full(101, Da_test)
ni_arr  = np.full(101, ni_test)
lam0_sq_code = compute_lambda0(r_grid, h, Da_arr, ni_arr)
print(f"  lambda0_sq from code = {lam0_sq_code:.6f}")
print(f"  lambda0   from code  = {lam0_sq_code**0.5:.6f}")
if abs(lam0_sq_code - lambda_analytic_sq) < 0.01 * lambda_analytic_sq:
    print("  -> CODE CORRECT: lambda0_sq = mu")
elif abs(lam0_sq_code - 1.0/lambda_analytic_sq) < 0.01 / lambda_analytic_sq:
    print("  -> BUG CONFIRMED: code computes lambda0_sq = 1/mu (inverted!)")
else:
    print(f"  -> UNEXPECTED: analytic={lambda_analytic_sq:.4f}, code={lam0_sq_code:.4f}")

# ─────────────────────────────────────────────────────────
# ПРОВЕРКА 3: H(r) — плоский при всех n_e?
# ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. H(r)/H(R): почему модель даёт плоский профиль?")
print("   Анализ уравнения для |H|^2 при малой проводимости")
print("=" * 60)
# При delta >> R, уравнение (1/r)d/dr[r*alpha*d|H|^2/dr] = 2*sigma_a*|E|^2
# alpha = sigma_a/|sigma|^2 ~ 1/sigma_p (при sigma_a << sigma_p)
# sigma_p = ne*e^2*omega / (me*(nu_c^2+omega^2)) ~ ne*e^2/(me*omega) когда nu_c>>omega
# при nu_c >> omega: sigma_a << sigma_p, alpha ~ sigma_a/sigma_p^2 ~ nu_c^2/omega^2 * sigma_a/sigma_a^2/...
# На самом деле проверим численно:
for ne in [3.71e17, 3.875e20]:
    sa, sp, sm2 = conductivity(ne, p_pa)
    alpha = sa / sm2
    E_wall_v = omega * mu0 * cfg.H_WALL * R / 2
    E_sq     = E_wall_v**2 / 2   # avg |E|^2 in tube
    rhs_est  = 2 * sa * E_sq
    # LHS scale: alpha * H_wall^2 / R^2
    lhs_scale = alpha * cfg.H_WALL**2 / R**2
    print(f"  n_e={ne:.2e}: sigma_a={sa:.3e}, alpha={alpha:.3e}")
    print(f"    LHS scale ~ alpha*H^2/R^2 = {lhs_scale:.3e}")
    print(f"    RHS       ~ 2*sigma_a*E^2 = {rhs_est:.3e}")
    print(f"    RHS/LHS   = {rhs_est/lhs_scale:.4f}  (>>1 => strong driving, flat H)")
    print()

# ─────────────────────────────────────────────────────────
# ПРОВЕРКА 4: Для каждого n_e какой H_wall даёт lambda0~1?
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("4. Какой H_wall нужен для lambda0=1 при экспериментальных n_e?")
print("   Сравнение с реальными экспериментальными условиями")
print("=" * 60)
from scipy.optimize import brentq

def find_Hwall_for_lam1(ne_target, p=133.0):
    """Найти H_wall при котором lambda0=1 для данного n_e."""
    def f(H_wall):
        res = solve_maxwell_for_ne0(n_e0=ne_target, R=R, p_pa=p, H_wall=H_wall)
        return res['lambda0'] - 1.0
    # Сначала проверим знаки
    lo, hi = 1e3, 1e6
    try:
        f_lo = f(lo)
        f_hi = f(hi)
        if f_lo * f_hi > 0:
            return None, None
        H_opt = brentq(f, lo, hi, xtol=100)
        res = solve_maxwell_for_ne0(n_e0=ne_target, R=R, p_pa=p, H_wall=H_opt)
        r_g  = res['r']
        H_abs = np.sqrt(np.maximum(res['u'], 0.0))
        H_norm0 = H_abs[0] / H_abs[-1] if H_abs[-1] > 0 else 1.0
        return H_opt, H_norm0
    except Exception as e:
        return None, None

for ne, label in [(3.71e17, "P=1.5 kW"), (7.83e17, "P=2.5 kW"), (22.4e17, "P=3.2 kW")]:
    H_opt, H0R = find_Hwall_for_lam1(ne)
    if H_opt is not None:
        E_opt = omega * mu0 * H_opt * R / 2
        sa, _, _ = conductivity(ne, p_pa)
        delta = (2/(omega*mu0*sa))**0.5
        print(f"  n_e={ne:.2e} ({label}): H_wall={H_opt:.0f} A/m,  E_wall={E_opt:.0f} V/m,  delta/R={delta/R:.1f},  H(0)/H(R)={H0R:.4f}")
    else:
        print(f"  n_e={ne:.2e} ({label}): no solution found")

print()
print("Эксперимент Рис.2.27: H(0)/H(R) = 0.760")
