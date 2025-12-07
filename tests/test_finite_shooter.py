import sys
import os
import numpy as np

# Add universe to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.planning import refine_finite_transfer
from universe.mission import FlightController

def test_finite_shooter_identity():
    """
    Test 1: Identity/Validation
    1. Define specific burn (e.g. 500m/s prograde) and propagate to get Target.
    2. Run Shooter starting from an offset guess.
    3. Verify Shooter recovers the original burn.
    """
    print("\n=== Finite Burn Shooter Whitebox Test ===")
    engine = PhysicsEngine()
    
    # Setup
    t0 = "2025-02-24T00:00:00Z"
    p, v = engine.get_body_state('ganymede', t0)
    
    # Avoid singularity (Center of Ganymede)
    p[0] += 3000.0 # 3000 km radius
    v_circ = np.sqrt(engine.GM['ganymede'] / 3000.0)
    v[1] += v_circ
    
    state_0 = list(p) + list(v)
    mass = 1000.0
    thrust = 2000.0
    isp = 3000.0
    
    # 1. Define True Burn
    # Arbitrary DV vector (Prograde + Out of plane)
    dv_target_mag = 0.5 # km/s
    burn_dir_true = np.array([0.8, 0.5, 0.3])
    burn_dir_true /= np.linalg.norm(burn_dir_true)
    
    # Duration
    g0 = 9.80665
    ve = isp * g0 / 1000.0
    m1 = mass * np.exp(-dv_target_mag / ve)
    duration = (mass - m1) * (ve * 1000.0) / thrust
    
    print(f"True Burn: DV={dv_target_mag} km/s, Dir={burn_dir_true}, Dur={duration:.2f} s")
    
    thrust_vec_true = burn_dir_true * thrust
    
    # Propagate True Burn
    burn_end_state, _ = engine.propagate_controlled(
        state_0, t0, duration, thrust_vec_true.tolist(), mass, isp
    )
    
    # Propagate Coast (6 hours)
    coast_dt = 21600.0 # 6 hours
    from datetime import datetime, timedelta
    t0_dt = datetime.fromisoformat(t0.replace('Z', '+00:00'))
    t_burn_end = t0_dt + timedelta(seconds=duration)
    t_burn_end_iso = t_burn_end.isoformat().replace('+00:00', 'Z')
    
    print(f"Burn End State: {burn_end_state}")
    print(f"Coast Start Time: {t_burn_end_iso}")
    print(f"Coast Duration: {coast_dt}")
    
    coast_res = engine.propagate(burn_end_state, t_burn_end_iso, coast_dt, t_eval=[0.0, coast_dt])
    print(f"Coast Result Len: {len(coast_res)}")
    r_target_final = np.array(coast_res[-1][:3])
    print(f"Target Position: {r_target_final}")
    
    # Define T_end for Shooter (t0 + duration + coast)
    t_end_dt = t_burn_end + timedelta(seconds=coast_dt)
    t_end_iso = t_end_dt.isoformat().replace('+00:00', 'Z')
    
    # 2. Run Shooter
    dv_true_vec = burn_dir_true * dv_target_mag
    
    print("Running Shooter...")
    
    # Pass Seed DV (True + Perturbation) to simulate realistic good impulsive guess
    seed_guess = dv_true_vec + np.array([0.05, -0.05, 0.02]) # 50-70 m/s error
    print(f"Seed Guess: {seed_guess} (Mag: {np.linalg.norm(seed_guess):.4f})")
    
    dv_sol = refine_finite_transfer(
        engine, 
        np.array(p), np.array(v), 
        t0, t_end_iso, 
        r_target_final, 
        mass, thrust, isp,
        seed_dv_vec=seed_guess
    )
    
    # 3. Validation
    # Compare dv_sol (Calculated Finite DV) with True Burn DV (Impulsive Equivalent?)
    # No, we defined True Burn as a Finite Burn.
    # So dv_sol should match `burn_dir_true * dv_target_mag`.
    
    dv_true_vec = burn_dir_true * dv_target_mag
    
    diff = dv_sol - dv_true_vec
    diff_mag = np.linalg.norm(diff)
    print(f"Solution DV: {dv_sol} (Mag: {np.linalg.norm(dv_sol):.4f})")
    print(f"True DV:     {dv_true_vec} (Mag: {np.linalg.norm(dv_true_vec):.4f})")
    print(f"DV Error:    {diff_mag*1000:.2f} m/s")
    
    if diff_mag < 0.01: # 10 m/s tolerance (fairly loose but acceptable for first pass)
        print("SUCCESS: Converged on True Burn.")
    else:
        print("FAILURE: Did not converge.")

if __name__ == "__main__":
    test_finite_shooter_identity()
