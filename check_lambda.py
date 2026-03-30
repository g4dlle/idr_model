import sys, os
sys.path.insert(0, os.path.join('idr_model'))

from self_consistent import solve_maxwell_for_ne0
import numpy as np

with open('lambda_results.txt', 'w') as f:
    for ne in [3e20, 5e20, 7e20, 1e21]:
        res = solve_maxwell_for_ne0(ne, N=60, R=0.012, p_pa=133.0, H_wall=100000.0,
                                     max_iter=1000, tol=1e-5, relax=0.3)
        f.write(f"ne={ne:.0e}: lambda0={res['lambda0']:.6f}  n_iter={res['n_iter']}  conv={res['converged']}  u_ratio={res['u'][0]/res['u'][-1]:.6f}\n")
print("Done")
