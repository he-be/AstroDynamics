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

def run_scenario():
    print("=== US-07: Ganymede to Callisto Transfer (Optimized) ===")
    
    engine = PhysicsEngine()
    optimizer = PorkchopOptimizer(engine)
    planner = MissionPlanner(engine)
    controller = FlightController(engine)
    
    # 1. OPTIMIZATION PHASE
    print("\n[Phase 1: Launch Window Optimization]")
    
    t_start_search = "2025-01-01T00:00:00Z"
    search_duration_days = 180.0 # 6 months
    
    # Parking Orbits (Alt ~200km)
    r_gan = 2634.0
    r_cal = 2410.0
    alt_park = 200.0
    r_park_dep = r_gan + alt_park
    r_park_arr = r_cal + alt_park
    
    print(f"Searching 6 month window from {t_start_search}...")
    print("Optimization Criteria: Min Delta V (Departure + Arrival)")
    
    # Coarse search to save time in demo
    # Step 2 days. TOF 2-16 days.
    best_dv, best_params = optimizer.optimize_window(
        'ganymede', 'callisto',
        t_start_search,
        window_duration_days=search_duration_days,
        flight_time_range_days=(2.0, 16.0),
        r_park_dep=r_park_dep,
        r_park_arr=r_park_arr,
        step_days=2.0,
        dt_step_days=1.0 
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
    
    # Re-calculate specific vectors for MissionPlanner
    # Get State of Ganymede at Launch
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch)
    
    # Initial State: Parking Orbit
    # We define ship state relative to Jupiter
    # Position: Ganymede + r_park (in direction of velocity? or radial?)
    # Ideally should align with departure hyperbola asymptote.
    # For demo, we start "at Ganymede" offset by parking radius in X-direction?
    # Or just start at p_gan + offset.
    # Note: Optimization assumed optimal departure burn from periapsis of hyperbolic escape.
    # Simply adding V_inf to V_gan isn't physically starting "in LEO".
    # But solving the spiral out is too complex.
    # We will simulate "Impulsive Departure from LEO" by:
    # 1. Calculating V_inf vector (Lambert).
    # 2. Calculating Delta V needed from Circular LEO.
    # 3. Assuming we perform this burn "instantaneously" at the right Perijove.
    # 4. Resulting State = Ganymede State + V_inf? 
    #    No, Resulting State = Ganymede State + V_inf is effectively being "at infinity" relative to Ganymede immediately.
    # This approximates the post-escape condition.
    # For N-body, starting exactly at Gan center is singular.
    # Starting at r_park requires integrating the escape hyperbola.
    # Simplification for Demo:
    # Start at Distance ~ SOI or slightly outside, with V = V_gan + V_inf?
    # Or Start at r_park, verify escape?
    # Finite burn escape is tricky.
    
    # Approach for "System Level Demo":
    # Start at: p_gan + (V_inf direction * r_park)? No.
    # Let's Start at: p_gan + Offset (e.g. 5000 km).
    # Velocity: v_gan + V_inf.
    # Effectively we skip the "Spiral/Hyperbola" and start on the Transfer trajectory.
    # The Delta V cost IS calculated correctly (Oberth included in optimization).
    # We just "pay" for it and spawn on the escape asymptote.
    
    dt_sec = dt_days * 86400.0
    t_arr_obj = datetime.fromisoformat(t_launch.replace('Z', '+00:00')) + timedelta(days=dt_days)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    
    p_cal, v_cal = engine.get_body_state('callisto', t_arr_iso)
    
    # Lambert
    v_dep_req, v_arr_pred = planner.calculate_transfer(p_gan, p_cal, dt_sec)
    
    # Vectors relative to moons
    v_inf_dep = v_dep_req - np.array(v_gan)
    v_inf_arr = v_arr_pred - np.array(v_cal)
    
    # Calculate Magnitude of maneuvers (Check consistency with Optimizer)
    v_inf_dep_mag = np.linalg.norm(v_inf_dep)
    v_inf_arr_mag = np.linalg.norm(v_inf_arr)
    
    mu_gan = engine.GM['ganymede']
    mu_cal = engine.GM['callisto']
    
    v_esc_dep = np.sqrt(v_inf_dep_mag**2 + 2*mu_gan/r_park_dep)
    v_circ_dep = np.sqrt(mu_gan/r_park_dep)
    dv1_mag = v_esc_dep - v_circ_dep # Oberth burn magnitude
    
    v_capt_arr = np.sqrt(v_inf_arr_mag**2 + 2*mu_cal/r_park_arr)
    v_circ_arr = np.sqrt(mu_cal/r_park_arr)
    dv2_mag = v_capt_arr - v_circ_arr
    
    print(f"  V_inf Dep: {v_inf_dep_mag:.4f} km/s. DV1: {dv1_mag:.4f} km/s")
    print(f"  V_inf Arr: {v_inf_arr_mag:.4f} km/s. DV2: {dv2_mag:.4f} km/s")
    print(f"  Total DV: {dv1_mag + dv2_mag:.4f} km/s (Matches Optimizer: {np.isclose(dv1_mag+dv2_mag, best_dv, rtol=1e-3)})")
    
    specs = {'mass': 1000.0, 'thrust': 2000.0, 'isp': 3000.0} # Nuclear
    
    is_ok, fuel = planner.verify_fuel(dv1_mag + dv2_mag, specs['mass'], specs['isp'])
    print(f"  Fuel Est: {fuel:.2f} kg")
    
    # 3. EXECUTION
    # To execute this logic in `propagate_controlled` (which propagates Heliocentric/Jovicentric):
    # We must start the integrator state.
    # IF we simulate the actual escape burn from LEO, we need very small time steps and Ganymede Gravity.
    # The current engine handles it (N-body).
    # BUT finding the correct firing angle in LEO to hit the V_inf asymptote is an optimization problem itself (Injection).
    # Simplification:
    # We execute "Deep Space Burns" equivalent to the V_inf change?
    # No, that ignores Oberth benefit.
    # We execute "Impulsive Approximation" by spawning on the transfer orbit?
    # AND subtracting fuel.
    
    # Let's simulate the "Transfer Phase" only.
    # Start Point: p_gan + (small offset).
    # Start Vel: v_dep_req.
    # This means we are "Already Escaped".
    # Fuel: We manually subtract `fuel_dep` from mass before starting.
    # And we simulate capture burn at end?
    # Or just fly by.
    
    # For a visually satisfying demo:
    # 1. Start at Ganymede.
    # 2. "Execute Departure" (Simulated by spawning with v_dep_req and reducing mass).
    # 3. Coast.
    # 4. MCC.
    # 5. Arrive Callisto.
    # 6. "Execute Arrival" (Reduce mass, check logic).
    
    print("\n[Phase 3: Execution]")
    
    # Setup Controller
    # Start at safe distance (40000 km) to ensure clean Lambert transfer.
    # We will back-propagate for visualization of the escape.
    offset_dist = 40000.0 # km
    
    # Direction: V_inf direction
    offset_dir = v_inf_dep / np.linalg.norm(v_inf_dep)
    
    r_start_sim = np.array(p_gan) + offset_dir * offset_dist
    
    # RE-SOLVE Lambert 
    print(f"  Adjusting forward start point to {offset_dist} km...")
    v_start_sim, _ = planner.calculate_transfer(r_start_sim, p_cal, dt_sec)
    
    # --- VISUALIZATION FILL ---
    # Back-propagate from start point to visualize the escape leg from Gravitational Well
    print("  Back-propagating escape leg for visualization...")
    # Backward propagation: flip velocity, go back in time
    # Engine requires positive time step usually? propagate(duration)
    # If duration is negative? Heyoka supports negative time? Yes.
    # But let's verify. propagate() method takes duration.
    # If duration negative, t_eval should be negative.
    
    back_state_0 = np.concatenate([r_start_sim, v_start_sim]).tolist()
    d_back = -0.5 * 86400 # 12 hours back
    t_eval_back = np.linspace(0, d_back, 50)
    
    try:
        # We propagate backwards from t_launch
        states_back = engine.propagate(back_state_0, t_launch, d_back, t_eval=t_eval_back)
        # Check radial distance to Ganymede
        traj_back = []
        for s in states_back:
            # Check distance to Ganymede at that time?
            # We are going back.
            # Just append. We'll clip if needed visually, but physics is truth.
            traj_back.append(s[:3])
            
        # Reverse traj_back to be chronological (T-12h -> T0)
        traj_back = traj_back[::-1]
    except Exception as e:
        print(f"Back-prop failed: {e}")
        traj_back = []

    # --- FORWARD SIMULATION ---
    # Subtract Departure Fuel
    _, fuel_dep = planner.verify_fuel(dv1_mag, specs['mass'], specs['isp'])
    mass_after_launch = specs['mass'] - fuel_dep
    
    state0 = np.concatenate([r_start_sim, v_start_sim]).tolist()
    controller.set_initial_state(state0, mass_after_launch, t_launch)
    
    print(f"Starting execution.")
    
    # Coast & MCC
    time_total = dt_sec
    time_half = time_total * 0.5
    
    controller.coast(time_half)
    
    # MCC
    print("Performing MCC...")
    curr_pos = controller.get_position()
    curr_vel = controller.get_velocity()
    
    # Re-target Callisto
    t_now = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    t_target = datetime.fromisoformat(t_arr_iso.replace('Z', '+00:00'))
    dt_left = (t_target - t_now).total_seconds()
    
    v_mcc_req, _ = planner.calculate_transfer(curr_pos, p_cal, dt_left)
    dv_mcc = v_mcc_req - curr_vel
    
    controller.execute_burn(dv_mcc, specs['thrust'], specs['isp'], label="MCC")
    
    # Coast Rest
    t_now = datetime.fromisoformat(controller.time_iso.replace('Z', '+00:00'))
    dt_left = (t_target - t_now).total_seconds()
    controller.coast(dt_left)
    
    # Arrival Check
    final_pos = controller.get_position()
    dist_cal = np.linalg.norm(final_pos - np.array(p_cal))
    
    print(f"\nArrival at Callisto:")
    print(f"  Distance: {dist_cal:.2f} km")
    
    # --- VISUALIZATION ---
    print("\nGenerating Jovian System Plot...")
    traj_fwd = np.array(controller.trajectory_log)
    
    # Combine Back + Fwd
    if len(traj_back) > 0:
        traj_combined = np.concatenate([traj_back, traj_fwd])
    else:
        traj_combined = traj_fwd
        
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # 1. Plot Trajectory
    ax.plot(traj_combined[:,0], traj_combined[:,1], label='Spacecraft', color='lime', linewidth=1.5)
    
    # 2. Plot Jupiter
    ax.scatter([0], [0], color='orange', s=200, label='Jupiter')
    
    # 3. Plot Moons (Io, Europa, Ganymede, Callisto)
    # Trace their orbits over the flight duration
    # We can get states at start and end, and maybe some mid points using engine?
    # Or just propagate analytical circles? No, Engine is N-body.
    # We can ask engine for states.
    moons = ['io', 'europa', 'ganymede', 'callisto']
    colors = {'io': 'yellow', 'europa': 'brown', 'ganymede': 'gray', 'callisto': 'cyan'}
    
    time_points = np.linspace(0, dt_days * 86400, 50)
    start_dt = datetime.fromisoformat(t_launch.replace('Z', '+00:00'))
    
    for moon in moons:
        moon_x = []
        moon_y = []
        for tp in time_points:
            t = (start_dt + timedelta(seconds=tp)).isoformat().replace('+00:00', 'Z')
            p, _ = engine.get_body_state(moon, t)
            moon_x.append(p[0])
            moon_y.append(p[1])
            
        ax.plot(moon_x, moon_y, color=colors[moon], linestyle=':', alpha=0.6)
        # Plot final position
        ax.scatter([moon_x[-1]], [moon_y[-1]], color=colors[moon], label=moon.capitalize(), s=50)
        # Plot start position
        ax.scatter([moon_x[0]], [moon_y[0]], color=colors[moon], marker='x', alpha=0.5)

    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#050510') # Space background
    plt.xlabel("Jovicentric X (km)")
    plt.ylabel("Jovicentric Y (km)")
    plt.title(f"US-07: Ganymede -> Callisto (Launch: {t_launch})")
    plt.legend(loc='upper right')
    
    plt.savefig('scenario_gan_cal.png')
    print("Saved scenario_gan_cal.png")

if __name__ == "__main__":
    run_scenario()
