import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add parent directory (universe) for legacy imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from universe.engine import PhysicsEngine
from universe.optimization import PorkchopOptimizer
from universe.mission import MissionPlanner, FlightController
from universe.planning import refine_transfer, refine_finite_transfer
from universe import frames
from universe import telemetry

def execute_tangential_burn(controller, engine, total_duration, thrust, isp, body_name):
    """
    Executes a burn while steering tangential to the velocity relative to the body.
    Segments the burn to update direction (Prograde Steering).
    """
    dt_step = 50.0 # 50 second steps (approx 2 deg arc)
    steps = int(total_duration / dt_step)
    remainder = total_duration - steps * dt_step
    
    print(f"  [Steering] Segmenting burn into {steps+1} steps (~{dt_step}s)...")
    
    for i in range(steps + 1):
        dt = dt_step if i < steps else remainder
        if dt < 1e-3: continue
        
        # Current State
        v_sc = np.array(controller.get_velocity())
        _, v_body = engine.get_body_state(body_name, controller.time_iso)
        v_rel = v_sc - np.array(v_body)
        
        # Steering Direction (Prograde)
        thrust_dir = v_rel / np.linalg.norm(v_rel)
        
        # Calculate Delta V for this step (Tsiolkovsky)
        g0 = 9.80665
        ve = isp * g0 / 1000.0 # km/s
        m0 = controller.mass
        # m_dot = F / (Isp * g0)
        # thrust in N, ve in m/s
        ve_m_s = isp * 9.80665
        m_dot = thrust / ve_m_s
        
        m1 = m0 - m_dot * dt
        if m1 <= 0:
            print("  [Error] Fuel exhausted!")
            break
            
        dv_step_km_s = ve * np.log(m0 / m1)
        dv_vec = thrust_dir * dv_step_km_s
        
        # Execute (Silent-ish? execute_burn prints)
        # We use a label that indicates progress
        controller.execute_burn(dv_vec, thrust, isp, label=f"Stp {i+1}")

def execute_gan_cal_mission(engine, t_launch, dt_days):
    print(f"\n[Execution] Launch: {t_launch}, Flight: {dt_days} days")
    controller = FlightController(engine)
    planner = MissionPlanner(engine)
    specs = {'mass': 1000.0, 'thrust': 2000.0, 'isp': 3000.0}
    
    dt_sec = dt_days * 86400.0
    t_arr_obj = datetime.fromisoformat(t_launch.replace('Z', '+00:00')) + timedelta(days=dt_days)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    
    # 1. Setup Parking Orbit Logic
    mu_gan = engine.GM['ganymede']
    r_gan = 2634.0
    alt_park = 200.0
    r_park_dep = r_gan + alt_park
    r_cal = 2410.0
    r_park_arr = r_cal + alt_park
    
    T_park = 2 * np.pi * np.sqrt(r_park_dep**3 / mu_gan)
    
    # Calculate Beta Angle for Parking Orbit Orientation
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch)
    p_cal, v_cal = engine.get_body_state('callisto', t_arr_iso)
    
    # Lambert for V_inf
    v_dep_lambert, _ = planner.calculate_transfer(p_gan, p_cal, dt_sec)
    v_inf_vec = v_dep_lambert - np.array(v_gan)
    v_inf_mag = np.linalg.norm(v_inf_vec)
    
    e_hyp = 1.0 + (r_park_dep * v_inf_mag**2) / mu_gan
    if e_hyp < 1.0: e_hyp = 1.0001
    beta = np.arccos(1.0 / e_hyp)
    
    # Orientation
    v_inf_hat = v_inf_vec / v_inf_mag
    n_vec = np.array([0., 0., 1.])
    v_perp = np.cross(n_vec, v_inf_hat)
    v_burn_dir = v_inf_hat * np.cos(beta) - v_perp * np.sin(beta)
    v_burn_dir /= np.linalg.norm(v_burn_dir)
    r_burn_dir = np.cross(n_vec, v_burn_dir)
    r_burn_dir /= np.linalg.norm(r_burn_dir)
    
    # Initial State at Burn Time (Virtual)
    r_rel_burn = r_burn_dir * r_park_dep
    v_circ = np.sqrt(mu_gan / r_park_dep)
    v_rel_burn = v_burn_dir * v_circ
    
    state_burn_jup = np.concatenate([
        np.array(p_gan) + r_rel_burn,
        np.array(v_gan) + v_rel_burn
    ])
    
    # Back-propagate to Start
    t_start_lag = T_park * 1.5
    t_sim_start_obj = datetime.fromisoformat(t_launch.replace('Z', '+00:00')) - timedelta(seconds=t_start_lag)
    t_sim_start = t_sim_start_obj.isoformat().replace('+00:00', 'Z')
    
    # Initialize Controller
    try:
        back_res = engine.propagate(state_burn_jup.tolist(), t_launch, -t_start_lag)
        initial_state = back_res
    except Exception:
        initial_state = state_burn_jup.tolist()

    controller.set_initial_state(initial_state, specs['mass'], t_sim_start)
    
    # --- COAST & FINITE BURN SEQ ---
    
    # A. Estimate Duration & Shift
    g0 = 9.80665
    ve = specs['isp'] * g0
    # Impulsive Guess from Lambert
    v_dep_lamb, _ = planner.calculate_transfer(p_gan, p_cal, dt_sec)
    dv_imp = v_dep_lamb - np.array(v_gan)
    dv_imp_mag = np.linalg.norm(dv_imp)
    dv_m = dv_imp_mag * 1000.0
    est_duration = (specs['mass'] * ve * (1 - np.exp(-dv_m/ve))) / specs['thrust']
    
    t_shift = -est_duration / 2.0
    print(f"  [Planning] Est. Duration: {est_duration:.1f} s. Timing Shift: {t_shift:.1f} s")
    
    # B. Coast to Burn Start
    # We started at t_launch - t_start_lag
    # We want to start burn at t_launch + t_shift
    # coast_duration = t_start_lag + t_shift
    
    coast_dur = t_start_lag + t_shift
    if coast_dur > 0:
        print(f"  [Coast] Coasting to Burn Start ({coast_dur/60:.1f} min)...")
        controller.coast(coast_dur, step_points=250)
        
    # C. Refine Finite Burn
    print("  [Maneuver] Refining Solution (Finite Shooter)...")
    
    # Current State (at burn start)
    state_start = controller.state
    r_start = state_start[:3]
    v_start = state_start[3:6]
    
    dv_sol = refine_finite_transfer(
        engine, r_start, v_start, 
        controller.time_iso, t_arr_iso, 
        list(p_cal), 
        controller.mass, specs['thrust'], specs['isp'],
        seed_dv_vec=dv_imp
    )
    
    # D. Execute
    controller.execute_burn(dv_sol, specs['thrust'], specs['isp'], label="Finite Shooter Burn")
    
    # Coast to Arrival
    print("  [Coast] Transit to Callisto...")
    t_now = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    dt_left = (t_arr_obj - t_now).total_seconds()
    controller.coast(dt_left, step_points=300)
    
    # Arrival Check
    final_pos = controller.get_position()
    p_cal_end, _ = engine.get_body_state('callisto', controller.time_iso)
    dist = np.linalg.norm(final_pos - np.array(p_cal_end))
    print(f"  [Result] Arrival Error: {dist:.1f} km")
    
    # Insertion Burn (Optional for metric, mostly for vis)
    v_circ_mag = np.sqrt(engine.GM['callisto'] / r_park_arr)
    curr_vel = controller.get_velocity()
    _, v_cal_now = engine.get_body_state('callisto', controller.time_iso)
    r_rel = final_pos - np.array(p_cal_end)
    v_rel = curr_vel - np.array(v_cal_now)
    h_vec = np.cross(r_rel, v_rel)
    h_hat = h_vec / np.linalg.norm(h_vec)
    v_circ_dir = np.cross(h_hat, r_rel)
    v_circ_dir = v_circ_dir / np.linalg.norm(v_circ_dir)
    v_target_state = np.array(v_cal_now) + v_circ_dir * v_circ_mag
    dv_insert = v_target_state - curr_vel
    controller.execute_burn(dv_insert, specs['thrust'], specs['isp'], label="Insertion Burn")
    
    T_dest = 2 * np.pi * np.sqrt(r_park_arr**3 / engine.GM['callisto'])
    controller.coast(T_dest)
    
    return dist, controller

def run_scenario():
    print("=== US-07: Ganymede to Callisto Transfer (Optimized) ===")
    
    engine = PhysicsEngine()
    optimizer = PorkchopOptimizer(engine)
    
    print("\n[Phase 1: Launch Window Optimization]")
    t_start_search = "2025-03-01T00:00:00Z"
    
    # Using previous optimal to save time, or research?
    # User asked for robustness test, so let's stick to the good date first.
    # But let's keep search for general usage.
    
    best_dv, best_params = optimizer.optimize_window(
        'ganymede', 'callisto',
        t_start_search,
        window_duration_days=180.0,
        flight_time_range_days=(2.0, 16.0),
        r_park_dep=2834.0,
        r_park_arr=2610.0,
        step_days=2.0,
        dt_step_days=1.0 
    )
    
    if best_params is None: return

    best_launch, best_dt = best_params
    print(f"\nOptimal Solution Found: {best_launch}, {best_dt} days")
    
    dist, controller = execute_gan_cal_mission(engine, best_launch, best_dt)
    
    # Visualization
    print("\nGenerating Jovian System Plot...")
    traj = np.array(controller.trajectory_log)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.plot(traj[:,0], traj[:,1], label='Spacecraft', color='lime', linewidth=1.5)
    ax.scatter([0], [0], color='orange', s=200, label='Jupiter')
    
    moons = ['io', 'europa', 'ganymede', 'callisto']
    colors = {'io': 'yellow', 'europa': 'cyan', 'ganymede': 'gray', 'callisto': 'red'}
    
    t0_iso = controller._trajectory_log[0]['time']
    tf_iso = controller._trajectory_log[-1]['time']
    start_dt = datetime.fromisoformat(t0_iso.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(tf_iso.replace('Z', '+00:00'))
    total_sec = (end_dt - start_dt).total_seconds()
    time_points = np.linspace(0, total_sec, 200)
    
    for moon in moons:
        moon_x = []
        moon_y = []
        for tp in time_points:
            t = (start_dt + timedelta(seconds=tp)).isoformat().replace('+00:00', 'Z')
            p, _ = engine.get_body_state(moon, t)
            moon_x.append(p[0])
            moon_y.append(p[1])
        ax.plot(moon_x, moon_y, color=colors[moon], linestyle=':', alpha=0.5)
        ax.scatter([moon_x[-1]], [moon_y[-1]], color=colors[moon], label=moon.capitalize())
        
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    plt.legend()
    plt.savefig("scenario_gan_cal.png")
    print("Saved scenario_gan_cal.png")
    
    telemetry.export_mission_manifest(controller, "scenario_gan_cal.json", mission_name="Ganymede-Callisto")

if __name__ == "__main__":
    run_scenario()
