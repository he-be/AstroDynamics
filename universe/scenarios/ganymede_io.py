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

    # 3. SETUP LOW ORBIT START
    mu_gan = engine.GM['ganymede']
    a_park = r_park_dep 
    T_park = 2 * np.pi * np.sqrt(a_park**3 / mu_gan)
    
    print(f"  Parking Orbit (Ganymede): Alt {alt_park}km, Period {T_park/60:.1f} min")
    
    # Target Setup State
    v_dep_req, v_arr_pred = planner.calculate_transfer(p_gan, p_io, dt_sec)
    v_inf_vec = v_dep_req - np.array(v_gan)
    
    # Hyperbolic Injection Geometry
    # Align periapsis burn with escape asymptote (V_inf)
    # Turning angle beta: cos(beta) = 1/e
    v_inf_mag = np.linalg.norm(v_inf_vec)
    e_hyp = 1.0 + (r_park_dep * v_inf_mag**2) / mu_gan
    beta = np.arccos(1.0 / e_hyp) 
    
    # V_burn direction: Rotate V_inf by -beta (Prograde/Left turn)
    v_inf_hat = v_inf_vec / v_inf_mag
    n_vec = np.array([0., 0., 1.])
    v_perp = np.cross(n_vec, v_inf_hat)
    
    v_burn_dir = v_inf_hat * np.cos(beta) - v_perp * np.sin(beta)
    v_burn_dir /= np.linalg.norm(v_burn_dir)
    
    # Position: r x v = h (Z) => r = cross(Z, v)
    r_burn_dir = np.cross(n_vec, v_burn_dir)
    r_burn_dir /= np.linalg.norm(r_burn_dir)
    
    r_rel_burn = r_burn_dir * r_park_dep
    v_circ = np.sqrt(mu_gan / r_park_dep)
    v_rel_burn = v_burn_dir * v_circ
    
    state_burn_jup = np.concatenate([
        np.array(p_gan) + r_rel_burn,
        np.array(v_gan) + v_rel_burn
    ])
    
    # BACK-PROPAGATE 1.2 Revs
    t_start_lag = T_park * 1.2
    t_sim_start_obj = datetime.fromisoformat(t_launch.replace('Z', '+00:00')) - timedelta(seconds=t_start_lag)
    t_sim_start = t_sim_start_obj.isoformat().replace('+00:00', 'Z')
    
    print(f"  Initializing Simulation at {t_sim_start} (T-{t_start_lag/60:.1f} min)")
    
    try:
        back_res = engine.propagate(state_burn_jup.tolist(), t_launch, -t_start_lag)
        initial_state = back_res 
    except Exception as e:
        print(f"  Back-prop failed, using analytical approx: {e}")
        initial_state = state_burn_jup.tolist()
    
    # Costs
    v_inf_dep_mag = np.linalg.norm(v_inf_vec)
    v_inf_arr_mag = np.linalg.norm(v_arr_pred - np.array(v_io))
    
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

    # 4. EXECUTION
    print("\n[Phase 3: Execution]")
    
    controller.set_initial_state(initial_state, specs['mass'], t_sim_start)
    
    # Coast 1 Rev
    print(f"  [Coast] Waiting for optimal window ({t_start_lag/60:.1f} min)...")
    controller.coast(t_start_lag)
    
    # DEPARTURE BURN
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    
    # 1. Get required V_inf vector from Lambert (Targeting Io)
    v_dep_lambert, _ = planner.calculate_transfer(curr_pos, p_io, dt_sec)
    p_gan_now, v_gan_now = engine.get_body_state('ganymede', t_launch)
    v_inf_target = v_dep_lambert - np.array(v_gan_now)
    v_inf_mag = np.linalg.norm(v_inf_target)
    
    # 2. Oberth Injection Calculation
    r_vec = curr_pos - np.array(p_gan_now)
    r_mag = np.linalg.norm(r_vec)
    v_inj_mag = np.sqrt(v_inf_mag**2 + 2 * mu_gan / r_mag)
    
    # 3. Burn Direction: Tangential (along current relative velocity)
    # We assume our Beta-angle setup aligned us correctly.
    v_rel = curr_vel - np.array(v_gan_now)
    v_rel_hat = v_rel / np.linalg.norm(v_rel)
    
    # Target State: V_gan + V_inj_mag * V_rel_hat
    v_target_inertial = np.array(v_gan_now) + v_rel_hat * v_inj_mag
    
    dv_burn = v_target_inertial - curr_vel
    dv_mag = np.linalg.norm(dv_burn)
    print(f"  [Maneuver] Executing Departure Burn (Est: {dv_mag*1000:.1f} m/s)")
    controller.execute_burn(dv_burn, specs['thrust'], specs['isp'], label="Departure Burn")
    
    # Coast
    controller.coast(dt_sec * 0.5)
    
    # MCC
    print("  [Maneuver] Performing MCC...")
    t_now_mcc = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    t_target = datetime.fromisoformat(t_arr_iso.replace('Z', '+00:00'))
    dt_left = (t_target - t_now_mcc).total_seconds()
    
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    v_req_mcc, _ = planner.calculate_transfer(curr_pos, p_io, dt_left)
    dv_mcc = v_req_mcc - curr_vel
    controller.execute_burn(dv_mcc, specs['thrust'], specs['isp'], label="MCC")
    
    # Coast to Arrival
    controller.coast(dt_left - 300)
    controller.coast(300)
    
    # ARRIVAL / INSERTION
    print("\n  [Arrival] Executing Insertion Burn...")
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    p_io_now, v_io_now = engine.get_body_state('io', controller.time_iso)
    
    r_rel = curr_pos - np.array(p_io_now)
    dist = np.linalg.norm(r_rel)
    print(f"    Distance to Io: {dist:.1f} km (Target: {r_park_arr:.1f} km)")
    
    v_circ_mag = np.sqrt(engine.GM['io'] / dist)
    
    v_rel = curr_vel - np.array(v_io_now)
    h_vec = np.cross(r_rel, v_rel)
    h_hat = h_vec / np.linalg.norm(h_vec)
    
    v_circ_dir = np.cross(h_hat, r_rel)
    v_circ_dir = v_circ_dir / np.linalg.norm(v_circ_dir)
    
    v_target_state = np.array(v_io_now) + v_circ_dir * v_circ_mag
    dv_insert = v_target_state - curr_vel
    
    controller.execute_burn(dv_insert, specs['thrust'], specs['isp'], label="Insertion Burn")
    
    # FINAL COAST
    T_dest = 2 * np.pi * np.sqrt(dist**3 / engine.GM['io'])
    print(f"  [Coast] Verifying stable orbit ({T_dest/60:.1f} min)...")
    controller.coast(T_dest)
    
    print("\nMission Complete.")

    # --- VISUALIZATION ---
    print("\nGenerating Jovian System Plot...")
    traj = np.array(controller.trajectory_log)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    ax.plot(traj[:,0], traj[:,1], label='Spacecraft', color='lime', linewidth=1.5)
    ax.scatter([0], [0], color='orange', s=200, label='Jupiter')
    
    moons = ['io', 'europa', 'ganymede']
    colors = {'io': 'yellow', 'europa': 'brown', 'ganymede': 'gray'}
    
    start_dt = t_sim_start_obj
    end_dt = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    total_sec = (end_dt - start_dt).total_seconds()
    
    time_points = np.linspace(0, total_sec, 100)
    
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
