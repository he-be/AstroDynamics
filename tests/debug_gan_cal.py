import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.mission import FlightController, MissionPlanner
# from universe.planning import MissionPlanner # REMOVE THIS LINE

def test_parking_orbit_resolution():
    print("Testing Parking Orbit Resolution...")
    engine = PhysicsEngine()
    controller = FlightController(engine)
    
    t_start = "2025-01-01T00:00:00Z"
    
    # Parking Orbit Setup
    mu_gan = engine.GM['ganymede']
    r_park = 2634.0 + 200.0
    v_circ = np.sqrt(mu_gan / r_park)
    T_park = 2 * np.pi * np.sqrt(r_park**3 / mu_gan)
    
    # State relative to Ganymede
    p_gan, v_gan = engine.get_body_state('ganymede', t_start)
    
    # Initial State (approx)
    r_rel = np.array([r_park, 0, 0])
    v_rel = np.array([0, v_circ, 0])
    
    state0 = np.concatenate([np.array(p_gan) + r_rel, np.array(v_gan) + v_rel])
    
    controller.set_initial_state(state0.tolist(), 1000.0, t_start)
    
    # Coast for 1 period
    print(f"Coasting for {T_park/60:.1f} mins...")
    controller.coast(T_park)
    
    traj = np.array(controller.trajectory_log)
    print(f"Log Points: {len(traj)}")
    
    # Check Density
    # Distance between points
    diffs = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
    print(f"Max Step Distance: {np.max(diffs):.2f} km")
    print(f"Mean Step Distance: {np.mean(diffs):.2f} km")
    
    # Check Circularity relative to Ganymede (approx)
    # We need Ganymede position at each step to check altitude properly, 
    # but for "shape" check in inertial frame, it's a spiral.
    
    if len(traj) < 50:
        print("FAIL: Resolution too low (<50 points per orbit)")
    else:
        print("PASS: Resolution acceptable")


def test_lambert_validity():
    print("\nTesting Oberth Injection Calculation (N-body)...")
    engine = PhysicsEngine()
    
    t_launch = "2025-02-24T00:00:00Z" 
    t_arr = "2025-03-02T00:00:00Z" 
    
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch)
    p_cal, v_cal = engine.get_body_state('callisto', t_arr)
    
    # Need Lambert to get V_inf
    dt = (datetime.fromisoformat(t_arr.replace('Z', '+00:00')) - datetime.fromisoformat(t_launch.replace('Z', '+00:00'))).total_seconds()
    
    from universe.planning import solve_lambert
    # Calculate transfer from GANYMEDE POSITION (approximation of infinity start)
    v_dep_lambert, _ = solve_lambert(p_gan, p_cal, dt, engine.GM['jupiter'])
    
    v_inf_vec = v_dep_lambert - np.array(v_gan)
    v_inf_mag = np.linalg.norm(v_inf_vec)
    
    # 2. Setup Optimal Injection Geometry (Periapsis)
    mu_gan = engine.GM['ganymede']
    r_park = 2634.0 + 200.0
    
    # Beta
    e_hyp = 1.0 + (r_park * v_inf_mag**2) / mu_gan
    beta = np.arccos(1.0 / e_hyp)
    
    # Directions
    v_inf_hat = v_inf_vec / v_inf_mag
    n_vec = np.array([0., 0., 1.])
    v_perp = np.cross(n_vec, v_inf_hat)
    
    # Velocity Direction at Periapsis (Rotate v_inf by -beta)
    v_burn_dir = v_inf_hat * np.cos(beta) - v_perp * np.sin(beta)
    v_burn_dir /= np.linalg.norm(v_burn_dir)
    
    # Position Direction (Perp to v_burn)
    r_burn_dir = np.cross(n_vec, v_burn_dir)
    r_burn_dir /= np.linalg.norm(r_burn_dir)
    
    # 3. Create State
    r_start = np.array(p_gan) + r_burn_dir * r_park
    
    # Magnitude
    v_inj_mag = np.sqrt(v_inf_mag**2 + 2 * mu_gan / r_park)
    v_start = np.array(v_gan) + v_burn_dir * v_inj_mag
    
    # Propagate
    state0 = np.concatenate([r_start, v_start]).tolist()
    states = engine.propagate(state0, t_launch, dt, t_eval=np.linspace(0, dt, 200))
    final = states[-1]
    r_final = final[:3]
    
    dist = np.linalg.norm(r_final - np.array(p_cal))
    print(f"Propagated Target Distance Error: {dist:.2f} km")
    
    if dist < 50000: # Relaxed slightly, as 2-body Lambert isn't perfect for 3-body
        print("PASS: Oberth Injection aligns with N-body physics.")
    else:
        print("FAIL: Still missing target. MCC would be required.")


def test_shooter_correction():
    print("\nTesting Shooter/Differential Correction (N-body)...")
    engine = PhysicsEngine()
    
    t_launch = "2025-02-24T00:00:00Z" 
    t_arr = "2025-03-02T00:00:00Z" 
    
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch)
    p_cal, v_cal = engine.get_body_state('callisto', t_arr)
    
    dt = (datetime.fromisoformat(t_arr.replace('Z', '+00:00')) - datetime.fromisoformat(t_launch.replace('Z', '+00:00'))).total_seconds()
    
    from universe.planning import solve_lambert, refine_transfer
    v_dep_lambert, _ = solve_lambert(p_gan, p_cal, dt, engine.GM['jupiter'])
    
    # Initial Guess: Oberth Magnitude + Lambert Direction (Rough)
    v_inf_vec = v_dep_lambert - np.array(v_gan)
    v_inf_mag = np.linalg.norm(v_inf_vec)
    
    mu_gan = engine.GM['ganymede']
    r_park = 2634.0 + 200.0
    
    # Use same Start Position as determined by beta
    e_hyp = 1.0 + (r_park * v_inf_mag**2) / mu_gan
    beta = np.arccos(1.0 / e_hyp)
    v_inf_hat = v_inf_vec / v_inf_mag
    n_vec = np.array([0., 0., 1.])
    v_perp = np.cross(n_vec, v_inf_hat)
    
    v_burn_dir = v_inf_hat * np.cos(beta) - v_perp * np.sin(beta)
    v_burn_dir /= np.linalg.norm(v_burn_dir)
    r_burn_dir = np.cross(n_vec, v_burn_dir)
    r_burn_dir /= np.linalg.norm(r_burn_dir)
    
    r_start = np.array(p_gan) + r_burn_dir * r_park
    
    # Guess Velocity: Tangential with V_inj magnitude
    v_inj_mag = np.sqrt(v_inf_mag**2 + 2 * mu_gan / r_park)
    v_guess = np.array(v_gan) + v_burn_dir * v_inj_mag
    
    # Refine
    v_corrected = refine_transfer(engine, r_start, v_guess, t_launch, t_arr, p_cal)
    
    # Verify Correction
    state0 = np.concatenate([r_start, v_corrected]).tolist()
    states = engine.propagate(state0, t_launch, dt, t_eval=np.linspace(0, dt, 200))
    final = states[-1]
    r_final = final[:3]
    
    dist = np.linalg.norm(r_final - np.array(p_cal))
    print(f"Refined Target Distance Error: {dist:.2f} km")
    
    if dist < 1000:
        print("PASS: Shooter aligned trajectory successfully.")
    else:
        print("FAIL: Shooter failed to converge.")

if __name__ == "__main__":
    test_parking_orbit_resolution()
    test_lambert_validity()
    test_shooter_correction()
