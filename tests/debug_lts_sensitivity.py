
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from universe.engine import PhysicsEngine

def test_lts_step():
    print("Initializing Engine...")
    engine = PhysicsEngine(include_saturn=False)
    
    r1 = np.array([-2646193.94528, 10464645.723, 44778.234])
    v1 = np.array([-10889.0, -2700.0, 100.0])
    
    # Target (Arbitrary, just to get an error)
    target_pos = r1 + np.array([1e6, 0, 0]) 
    
    state_vec = np.concatenate([r1, v1])
    time_str = "2023-01-01T00:00:00Z"
    
    params = np.array([-0.98, 0.14, 0.06, 0.0, 0.0, 0.0]) # From logs
    dt = 3600.0 * 24 * 4 # 4 days
    
    thrust = 1000.0
    
    print("--- 0. Constant Steering Pre-Run ---")
    engine.propagate_controlled(
        state_vec, time_str, dt, np.array([1.0, 0.0, 0.0]), 2000.0, 320.0,
        steering_mode='constant', thrust_magnitude=thrust
    )
    
    print("--- 1. Nominal Run ---")
    res, _ = engine.propagate_controlled(
        state_vec, time_str, dt, params, 2000.0, 320.0, 
        steering_mode='linear_tangent', thrust_magnitude=thrust
    )
    r_nom = np.array(res[0:3])
    miss = r_nom - target_pos
    err_0 = np.linalg.norm(miss)
    print(f"Error 0: {err_0}")
    
    print("--- 2. Jacobian ---")
    J = np.zeros((3, 6))
    eps = 1e-4
    
    for i in range(6):
        p_p = params.copy()
        p_p[i] += eps
        res_p, _ = engine.propagate_controlled(
            state_vec, time_str, dt, p_p, 2000.0, 320.0,
            steering_mode='linear_tangent', thrust_magnitude=thrust
        )
        r_p = np.array(res_p[0:3])
        col = (r_p - r_nom) / eps
        J[:, i] = col
    
    print(f"J norm: {np.linalg.norm(J)}")
    
    print("--- 3. Update ---")
    dx = -np.linalg.pinv(J) @ miss
    print(f"dx: {dx}")
    
    params_new = params + dx
    
    print("--- 4. New Run ---")
    res_new, _ = engine.propagate_controlled(
        state_vec, time_str, dt, params_new, 2000.0, 320.0, 
        steering_mode='linear_tangent', thrust_magnitude=thrust
    )
    r_new = np.array(res_new[0:3])
    err_1 = np.linalg.norm(r_new - target_pos)
    
    print(f"Error 1: {err_1}")
    print(f"Improvement: {err_0 - err_1}")

if __name__ == "__main__":
    test_lts_step()
