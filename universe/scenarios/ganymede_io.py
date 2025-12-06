import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import PhysicsEngine
from optimization import PorkchopOptimizer
from mission import MissionPlanner, FlightController
import frames
import telemetry

def run_scenario():
    print("=== US-07 Variant: Ganymede to Io Transfer (Direct) ===")
    
    engine = PhysicsEngine()
    optimizer = PorkchopOptimizer(engine)
    planner = MissionPlanner(engine)
    controller = FlightController(engine)
    
    # 1. OPTIMIZATION PHASE
    print("\n[Phase 1: Launch Window Optimization]")
    
    t_start_search = "2025-01-01T00:00:00Z"
    search_duration_days = 30.0 # Short window logic for Io (frequent opportunities)
    
    # Parking Orbits (Alt ~200km)
    r_gan = 2634.0
    r_io = 1821.0
    alt_park = 200.0
    r_park_dep = r_gan + alt_park
    r_park_arr = r_io + alt_park
    
    print(f"Searching 30 day window from {t_start_search}...")
    print("Optimization Criteria: Min Delta V (Departure + Arrival)")
    
    # Io transfer is faster. Search 0.5 to 5.0 days.
    best_dv, best_params = optimizer.optimize_window(
        'ganymede', 'io',
        t_start_search,
        window_duration_days=search_duration_days,
        flight_time_range_days=(0.5, 5.0),
        r_park_dep=r_park_dep,
        r_park_arr=r_park_arr,
        step_days=0.5, # Finer step for Io
        dt_step_days=0.1
    )
    
    if best_params is None:
        print("Optimization failed to find valid transfer.")
        return
        
    t_launch, dt_days = best_params
    print(f"\nOptimal Solution Found:")
    print(f"  Launch Date: {t_launch}")
    print(f"  Flight Time: {dt_days} days")
    print(f"  Est. Delta V: {best_dv:.4f} km/s")
    
    # 2. PLANNING PHASE
    print("\n[Phase 2: Detailed Mission Planning]")
    
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch)
    
    dt_sec = dt_days * 86400.0
    t_arr_obj = datetime.fromisoformat(t_launch.replace('Z', '+00:00')) + timedelta(days=dt_days)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    
    p_io, v_io = engine.get_body_state('io', t_arr_iso)
    
    # Lambert
    v_dep_req, v_arr_pred = planner.calculate_transfer(p_gan, p_io, dt_sec)
    
    # Vectors relative to moons
    v_inf_dep = v_dep_req - np.array(v_gan)
    v_inf_arr = v_arr_pred - np.array(v_io)
    
    v_inf_dep_mag = np.linalg.norm(v_inf_dep)
    v_inf_arr_mag = np.linalg.norm(v_inf_arr)
    
    mu_gan = engine.GM['ganymede']
    mu_io = engine.GM['io']
    
    v_esc_dep = np.sqrt(v_inf_dep_mag**2 + 2*mu_gan/r_park_dep)
    v_circ_dep = np.sqrt(mu_gan/r_park_dep)
    dv1_mag = v_esc_dep - v_circ_dep
    
    v_capt_arr = np.sqrt(v_inf_arr_mag**2 + 2*mu_io/r_park_arr)
    v_circ_arr = np.sqrt(mu_io/r_park_arr)
    dv2_mag = v_capt_arr - v_circ_arr
    
    print(f"  V_inf Dep: {v_inf_dep_mag:.4f} km/s. DV1: {dv1_mag:.4f} km/s")
    print(f"  V_inf Arr: {v_inf_arr_mag:.4f} km/s. DV2: {dv2_mag:.4f} km/s")
    print(f"  Total DV: {dv1_mag + dv2_mag:.4f} km/s")
    
    specs = {'mass': 1000.0, 'thrust': 2000.0, 'isp': 3000.0}
    
    # 3. EXECUTION
    print("\n[Phase 3: Execution]")
    
    # Start at safe distance for clean Lambert
    offset_dist = 20000.0 # km (Reduced for Io/Gan scale? Safe enough)
    
    # Direction: V_inf direction
    offset_dir = v_inf_dep / np.linalg.norm(v_inf_dep)
    
    r_start_sim = np.array(p_gan) + offset_dir * offset_dist
    
    # RE-SOLVE Lambert 
    print(f"  Adjusting forward start point to {offset_dist} km...")
    v_start_sim, _ = planner.calculate_transfer(r_start_sim, p_io, dt_sec)
    
    # Back-propagation
    print("  Back-propagating escape leg for visualization...")
    back_state_0 = np.concatenate([r_start_sim, v_start_sim]).tolist()
    d_back = -0.5 * 86400 
    t_eval_back = np.linspace(0, d_back, 50)
    
    try:
        states_back = engine.propagate(back_state_0, t_launch, d_back, t_eval=t_eval_back)
        traj_back = []
        for s in states_back:
            traj_back.append(s[:3])
        traj_back = traj_back[::-1]
    except Exception as e:
        print(f"Back-prop failed: {e}")
        traj_back = []

    # Forward Simulation
    _, fuel_dep = planner.verify_fuel(dv1_mag, specs['mass'], specs['isp'])
    mass_after_launch = specs['mass'] - fuel_dep
    
    state0 = np.concatenate([r_start_sim, v_start_sim]).tolist()
    controller.set_initial_state(state0, mass_after_launch, t_launch)
    
    # Coast & MCC
    controller.coast(dt_sec * 0.5)
    
    print("Performing MCC...")
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    # t_now_stats = controller.get_state() # removed invalid call
    
    t_now = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    t_target = datetime.fromisoformat(t_arr_iso.replace('Z', '+00:00'))
    dt_left = (t_target - t_now).total_seconds()
    
    v_mcc_req, _ = planner.calculate_transfer(curr_pos, p_io, dt_left)
    dv_mcc = v_mcc_req - curr_vel
    
    controller.execute_burn(dv_mcc, specs['thrust'], specs['isp'], label="MCC")
    
    # Coast Rest
    t_now = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    dt_left = (t_target - t_now).total_seconds()
    controller.coast(dt_left)
    
    final_pos = controller.get_position()
    dist_io = np.linalg.norm(final_pos - np.array(p_io))
    
    print(f"\nArrival at Io:")
    print(f"  Distance: {dist_io:.2f} km")
    
    # --- VISUALIZATION ---
    print("\nGenerating Jovian System Plot...")
    traj_fwd = np.array(controller.trajectory_log)
    
    if len(traj_back) > 0:
        traj_combined = np.concatenate([traj_back, traj_fwd])
    else:
        traj_combined = traj_fwd
        
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    ax.plot(traj_combined[:,0], traj_combined[:,1], label='Spacecraft', color='lime', linewidth=1.5)
    ax.scatter([0], [0], color='orange', s=200, label='Jupiter')
    
    moons = ['io', 'europa', 'ganymede'] # Focus on inner moons
    colors = {'io': 'yellow', 'europa': 'brown', 'ganymede': 'gray'}
    
    start_dt = datetime.fromisoformat(t_launch.replace('Z', '+00:00'))
    time_len_sec = dt_days * 86400.0
    
    # Use finer resolution for moons since Io moves fast
    time_points = np.linspace(0, time_len_sec, 100)
    
    for moon in moons:
        moon_x = []
        moon_y = []
        for tp in time_points:
            t = (start_dt + timedelta(seconds=tp)).isoformat().replace('+00:00', 'Z')
            p, _ = engine.get_body_state(moon, t)
            moon_x.append(p[0])
            moon_y.append(p[1])
            
        ax.plot(moon_x, moon_y, color=colors[moon], linestyle=':', alpha=0.6)
        ax.scatter([moon_x[-1]], [moon_y[-1]], color=colors[moon], label=moon.capitalize(), s=50)
        ax.scatter([moon_x[0]], [moon_y[0]], color=colors[moon], marker='x', alpha=0.5)

    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#050510')
    plt.xlabel("Jovicentric X (km)")
    plt.ylabel("Jovicentric Y (km)")
    plt.title(f"Ganymede -> Io Direct (Launch: {t_launch})")
    plt.legend(loc='upper right')
    
    plt.savefig('scenario_gan_io.png')
    plt.savefig('scenario_gan_io.png')
    print("Saved scenario_gan_io.png")
    
    # Export Telemetry
    telemetry.export_mission_manifest(controller, 'scenario_gan_io.json', mission_name="Ganymede-Io Direct")

if __name__ == "__main__":
    run_scenario()
