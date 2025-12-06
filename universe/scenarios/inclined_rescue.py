import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import PhysicsEngine
from mission import MissionPlanner, FlightController
from planning import solve_lambert, kepler_to_cartesian

def get_target_state(engine, t_iso):
    """
    Calculate state of Target Satellite involved in rescue.
    Orbit: Around Callisto.
    a = 5000 km. i = 45 deg. Circular.
    """
    # 1. Callisto State (Jovicentric)
    p_cal, v_cal = engine.get_body_state('callisto', t_iso)
    
    # 2. Satellite Orbit (Relative to Callisto)
    mu_cal = engine.GM['callisto']
    a = 5000.0 # km
    e = 0.0
    i_rad = np.radians(45.0)
    Omega_rad = np.radians(0.0)
    omega_rad = np.radians(0.0)
    
    # Mean Motion
    n = np.sqrt(mu_cal / a**3)
    
    # Reference Epoch: 2025-01-01
    # Ensure aware
    t_obj = datetime.fromisoformat(t_iso.replace('Z', '+00:00'))
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    dt_epoch = t_obj - epoch
    t_sec = dt_epoch.total_seconds()
    
    # Mean Anomaly
    M = n * t_sec
    nu = M # Circular
    
    r_rel, v_rel = kepler_to_cartesian(a, e, i_rad, Omega_rad, omega_rad, nu, mu_cal)
    
    # 3. Total State (Jovicentric)
    r_total = np.array(p_cal) + r_rel
    v_total = np.array(v_cal) + v_rel
    
    return r_total, v_total

def run_scenario():
    print("=== US-10: Inclined Rescue (Callisto 45-deg Orbit) ===")
    
    engine = PhysicsEngine()
    controller = FlightController(engine)
    planner = MissionPlanner(engine)
    
    # 1. OPTIMIZATION (Search for Node Crossing / Optimal Transfer)
    print("\n[Phase 1: Search for Optimal Intercept Window]")
    
    t_start_search = datetime(2025, 1, 1, tzinfo=timezone.utc)
    search_duration = 30.0 # days
    best_dv = float('inf')
    best_sol = None # (t_launch, t_arr, dt)
    
    # Grid Search
    # Launch: Every 1 day.
    # Flight Time: 1 to 10 days.
    # Check total Delta V (Dep from Ganymede LEO + Match Velocity at Target).
    # Since Target is in inclined orbit, matching velocity is expensive unless planes align.
    
    print(f"Scanning {int(search_duration)} days launch window...")
    for d_launch in range(int(search_duration)):
        t_launch = t_start_search + timedelta(days=d_launch)
        t_launch_iso = t_launch.isoformat().replace('+00:00', 'Z')
        
        # Ganymede State
        p_gan, v_gan = engine.get_body_state('ganymede', t_launch_iso)
        
        # Flight Time loop
        for dt_day in np.linspace(2.0, 10.0, 9): # 1 day steps
            t_arr = t_launch + timedelta(days=dt_day)
            t_arr_iso = t_arr.isoformat().replace('+00:00', 'Z')
            
            # Target State
            try:
                r_target, v_target = get_target_state(engine, t_arr_iso)
            
                # Lambert
                v_dep, v_arr = solve_lambert(np.array(p_gan), r_target, dt_day*86400, engine.GM['jupiter'])
            except Exception as e:
                print(f"Solver Error: {e}")
                continue
                
            # Delta Vs
            # Dep (Oberth from Ganymede)
            v_inf_dep = v_dep - np.array(v_gan)
            dv_dep = np.linalg.norm(v_inf_dep) # Simplified (Low Orbit)
            
            # Arr (Match Target Velocity directly)
            # This is "Impulsive Rendezvous".
            dv_arr = np.linalg.norm(np.array(v_target) - v_arr)
            
            total = dv_dep + dv_arr
            
            if total < best_dv:
                best_dv = total
                best_sol = (t_launch_iso, t_arr_iso, dt_day, dv_dep, dv_arr)
                
    if not best_sol:
        print("Optimization Failed.")
        return
        
    t_l, t_a, dt, dv1, dv2 = best_sol
    print(f"Optimal Solution:")
    print(f"  Launch: {t_l}")
    print(f"  Arrival: {t_a} (Flight: {dt} days)")
    print(f"  Total DV: {dv1+dv2:.4f} km/s (Dep: {dv1:.2f}, Arr: {dv2:.2f})")
    
    # 2. EXECUTION
    print("\n[Phase 2: Execution]")
    
    # Setup State
    p_gan, v_gan = engine.get_body_state('ganymede', t_l)
    r_target, v_target = get_target_state(engine, t_a)
    
    # Recalculate precision transfer
    dt_sec = dt * 86400.0
    v_dep_precise, v_arr_precise = solve_lambert(np.array(p_gan), r_target, dt_sec, engine.GM['jupiter'])
    
    # Start Offset
    v_inf_vec = v_dep_precise - np.array(v_gan)
    offset_dir = v_inf_vec / np.linalg.norm(v_inf_vec)
    r_start = np.array(p_gan) + offset_dir * 40000.0
    
    # Resolute
    v_start, v_arr_final = solve_lambert(r_start, r_target, dt_sec, engine.GM['jupiter'])
    
    # Launch
    state0 = np.concatenate([r_start, v_start]).tolist()
    controller.set_initial_state(state0, 1000.0, t_l)
    
    print("Executing Transfer coast...")
    controller.coast(dt_sec)
    
    # 3. INTERCEPT
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    
    diff_r = np.linalg.norm(curr_pos - r_target)
    print(f"Intercept Position Error: {diff_r:.2f} km")
    
    # Match Velocity (Rescue)
    dv_rescue = np.array(v_target) - curr_vel
    controller.execute_burn(dv_rescue, 2000.0, 3000.0, label="Rendezvous Burn")
    
    # 4. PLOT
    print("Generating Plot...")
    traj = np.array(controller.trajectory_log)
    
    plt.figure(figsize=(10,10))
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label='Rescue Traj', color='lime')
    
    # Plot Callisto and Target Orbit ring?
    # Callisto
    p_c, _ = engine.get_body_state('callisto', t_a)
    ax.scatter([p_c[0]], [p_c[1]], [p_c[2]], color='cyan', s=100, label='Callisto')
    
    # Target Orbit visual
    # Generate points around Callisto
    thetas = np.linspace(0, 2*np.pi, 50)
    for theta in thetas:
        # Kepler to cart needed again... or just manual ring
        pass 
        # Skip for simplicity, point intercept verified.
        
    ax.scatter([r_target[0]], [r_target[1]], [r_target[2]], color='red', marker='*', s=200, label='Target')
    
    ax.legend()
    plt.savefig('scenario_rescue.png')
    print("Saved scenario_rescue.png")

if __name__ == "__main__":
    run_scenario()
