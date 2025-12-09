
import sys
import os
import numpy as np
import time

# Add universe to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from universe.engine import PhysicsEngine

def test_lts_vareq():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    # Setup Scenario (Jupiter -> Callisto roughly)
    t_start = "2023-01-01T00:00:00Z"
    
    # State: [r, v] in SSB (km, km/s)
    # Just grab some valid state from engine (Jupiter relative)
    # We can use the same state as debug_lts_sensitivity
    r_start = np.array([-7.14201880e+08,  2.83980749e+08,  1.32626569e+08])
    v_start = np.array([-4.52697800e+00, -1.16873990e+01, -4.96541000e+00])
    
    state_vec = np.concatenate([r_start, v_start])
    
    dt = 3600.0 * 24 * 4 # 4 days
    
    # Control Params (random-ish)
    params = np.array([-0.8, 0.2, 0.1, 0.0, 0.0, 0.0])
    thrust = 10.0 # 10 N
    isp = 3000.0
    mass = 2000.0
    
    print("\n--- 1. Analytic Jacobian Run ---")
    t0 = time.time()
    res, m_final, jac = engine.propagate_controlled(
        state_vec, t_start, dt, params, mass, isp,
        steering_mode='linear_tangent', thrust_magnitude=thrust,
        with_variational_equations=True
    )
    t1 = time.time()
    
    print(f"Analytic Run Time: {t1-t0:.4f} s")
    print("Analytic Jacobian (Partial):")
    print(jac)
    
    if jac.shape != (3, 6):
        print(f"ERROR: Jacobian shape mismatch. Expected (3, 6), got {jac.shape}")
        sys.exit(1)
        
    print("\n--- 2. Finite Difference Jacobian Run ---")
    
    eps = 1e-5
    jac_fd = np.zeros((3, 6))
    
    # Nominal run for FD
    # Note: We must compare against nominal run without VarEq to ensure consistency, 
    # but with VarEq=True, the state should be identical.
    
    r_nom = np.array(res[0:3])
    
    # Use standard propagation for FD to test cross-consistency
    res_sanity, _ = engine.propagate_controlled(
        state_vec, t_start, dt, params, mass, isp,
        steering_mode='linear_tangent', thrust_magnitude=thrust,
        with_variational_equations=False
    )
    r_sanity = np.array(res_sanity[0:3])
    
    diff_sanity = np.linalg.norm(r_nom - r_sanity)
    print(f"Sanity Check (VarEq vs NoVarEq Position Diff): {diff_sanity:.6e} km")
    if diff_sanity > 1e-6:
        print("WARNING: VarEq trajectory diverges from standard trajectory!")
    
    t0_fd = time.time()
    for i in range(6):
        p_p = params.copy()
        p_p[i] += eps
        
        res_p, _ = engine.propagate_controlled(
            state_vec, t_start, dt, p_p, mass, isp,
            steering_mode='linear_tangent', thrust_magnitude=thrust,
            with_variational_equations=False
        )
        r_p = np.array(res_p[0:3])
        
        col = (r_p - r_nom) / eps
        jac_fd[:, i] = col
    t1_fd = time.time()
    
    print(f"Finite Diff Run Time: {t1_fd - t0_fd:.4f} s")
    print("Finite Diff Jacobian (Partial):")
    print(jac_fd)
    
    print("\n--- 3. Comparison ---")
    diff = jac - jac_fd
    print("Difference (Analytic - FD):")
    print(diff)
    
    norm_diff = np.linalg.norm(diff)
    print(f"Matrix Norm Difference: {norm_diff:.6e}")
    
    rel_error = norm_diff / (np.linalg.norm(jac_fd) + 1e-9)
    print(f"Relative Error: {rel_error:.6e}")
    
    if rel_error < 1e-2: # Relaxed tolerance for FD vs Analytic
        print("SUCCESS: Analytic Jacobian matches Finite Difference.")
    else:
        print("FAILURE: Jacobian mismatch.")
        sys.exit(1)

if __name__ == "__main__":
    test_lts_vareq()
