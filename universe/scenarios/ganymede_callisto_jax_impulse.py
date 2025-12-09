
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
from universe.mission import MissionPlanner, FlightController
from universe.jax_planning import JAXPlanner
from universe import telemetry

def run_jax_impulse_scenario():
    print("=== US-08: Ganymede to Callisto (JAX Impulse Demo) ===")
    
    # 1. Initialization
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    controller = FlightController(engine)
    
    # Mission Specs
    # Launch: 2025-03-10
    # Flight Time: ~4.0 days
    t_launch_iso = "2025-03-10T12:00:00Z"
    dt_days = 4.0
    dt_sec = dt_days * 86400.0
    
    t_arr_obj = datetime.fromisoformat(t_launch_iso.replace('Z', '+00:00')) + timedelta(days=dt_days)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    
    print(f"Launch: {t_launch_iso}")
    print(f"Arrival: {t_arr_iso}")
    
    # 2. Initial State Setup
    # Start at Ganymede (Offset by 5000km to emulate parking orbit / valid start)
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch_iso)
    
    # Simple explicit offset (e.g. 5000km altitude)
    r_start = np.array(p_gan) + np.array([5000.0, 0.0, 0.0])
    # Match Ganymede velocity exactly for start (Simulates being "detached" but co-moving)
    v_start = np.array(v_gan)
    
    controller.set_initial_state(
        np.concatenate([r_start, v_start]).tolist(),
        mass=1000.0,
        time_iso=t_launch_iso
    )
    
    # 3. Planning (JAX Impulse)
    print("\n[Planning] Calculating Impulsive Maneuver using JAXPlanner...")
    
    p_cal, _ = engine.get_body_state('callisto', t_arr_iso)
    
    # Solve for optimal initial velocity
    # Solve for optimal initial velocity
    t0 = datetime.now()
    
    # Calculate Lambert Guess for robust start
    from universe.planning import solve_lambert
    mu_jup = engine.GM['jupiter']
    v_lambert, _ = solve_lambert(np.array(r_start), np.array(p_cal), dt_sec, mu_jup)
    
    v_optimal = jax_planner.solve_impulsive_shooting(
        r_start=list(r_start),
        t_start_iso=t_launch_iso,
        dt_seconds=dt_sec,
        r_target=list(p_cal),
        initial_v_guess=list(v_lambert) 
    )
    t1 = datetime.now()
    print(f"  [Planner] Solution found in {(t1-t0).total_seconds():.2f}s")
    
    # Calculate Delta-V
    dv_vec = np.array(v_optimal) - v_start
    dv_mag = np.linalg.norm(dv_vec)
    print(f"  [Solution] Required Delta-V: {dv_mag*1000.0:.1f} m/s")
    print(f"  [Solution] Velocity Vector: {v_optimal}")
    
    # 4. Execution (Impulsive Burn)
    print("\n[Execution] Performing Impulsive Burn...")
    # FlightController supports 'execute_burn'. For impulse, we assume infinite thrust or just update velocity.
    # But execute_burn simulates finite burn.
    # For demonstration of "Impulse Solver", we can either:
    # A) Hack state directly (Perfect Impulse)
    # B) Use very high thrust short burn.
    # Let's use A for pure verification of the Solver's geometric accuracy.
    # But FlightController tracks history.
    
    # Let's do a "Perfect Impulse" via state update for this demo
    # ensuring we isolate the Solver accuracy from Finite Burn losses.
    current_state = np.array(controller.state)
    current_state[3:6] = v_optimal
    
    # Log the event manually or update controller
    controller.state = current_state.tolist()
    controller.trajectory_log.append({
        'time': t_launch_iso,
        'position': current_state[0:3].tolist(),
        'velocity': current_state[3:6].tolist(),
        'mass': 1000.0,
        'event': 'Impulsive Burn (JAX Solution)'
    })
    
    print("\n[Coast] Propagating to Arrival (JAX Engine)...")
    # Use JAX Evaluation for speed.
    jax_log = jax_planner.evaluate_trajectory(
        r_start=current_state[0:3].tolist(),
        v_start=current_state[3:6].tolist(),
        t_start_iso=t_launch_iso,
        dt_seconds=dt_sec,
        mass=1000.0,
        n_steps=100
    )
    
    # Update controller result for end checks
    final_entry = jax_log[-1]
    controller.time_iso = final_entry['time']
    # controller.state needs updating if we want get_position to work
    controller.state = final_entry['position'] + final_entry['velocity']
    
    # Append to log
    controller._trajectory_log.extend([{
        'time': e['time'], 
        'state': e['position'] + e['velocity'], 
        'mass': e['mass']
    } for e in jax_log])
    
    # 5. Result Verification
    final_pos = controller.get_position()
    p_cal_end, _ = engine.get_body_state('callisto', controller.time_iso)
    err = np.linalg.norm(final_pos - np.array(p_cal_end))
    
    print(f"\n[Result] Final Position Error: {err:.1f} km")
    
    if err < 20000.0: # <20,000 km is "Hit" for initial impulse without TCM
        print("PASS: Trajectory successfully intercepted Callisto vicinity.")
    else:
        print("WARNING: Trajectory deviation larger than expected.")
        
    # 6. Visualization
    print("\nGenerating Plot...")
    # trajectory_log property returns list of [x,y,z]
    traj = np.array(controller.trajectory_log)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.plot(traj[:,0], traj[:,1], label='Spacecraft', color='lime', linewidth=1.5)
    
    # Plot Bodies
    moons = ['ganymede', 'callisto']
    colors = {'ganymede': 'gray', 'callisto': 'red'}
    
    # Plot Start/End positions for moons
    for moon in moons:
        p0, _ = engine.get_body_state(moon, t_launch_iso)
        pf, _ = engine.get_body_state(moon, t_arr_iso)
        ax.scatter([p0[0]], [p0[1]], color=colors[moon], marker='o', label=f"{moon} (Start)")
        ax.scatter([pf[0]], [pf[1]], color=colors[moon], marker='x', label=f"{moon} (End)")
        
        # Approximate path
        p_mid, _ = engine.get_body_state(moon, (t_arr_obj - timedelta(days=dt_days/2)).isoformat())
        # ax.plot([p0[0], p_mid[0], pf[0]], [p0[1], p_mid[1], pf[1]], color=colors[moon], linestyle=':')

    # Highlight Target
    ax.scatter([p_cal_end[0]], [p_cal_end[1]], s=200, facecolors='none', edgecolors='red', linestyle='--')
    
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.grid(True, alpha=0.2)
    plt.legend()
    plt.title(f"JAX Impulse Demo: Error {err:.1f} km")
    plt.savefig("scenario_jax_impulse.png")
    print("Saved scenario_jax_impulse.png")

if __name__ == "__main__":
    run_jax_impulse_scenario()
