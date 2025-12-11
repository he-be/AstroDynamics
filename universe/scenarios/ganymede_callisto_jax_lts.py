
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import json

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def run_jax_lts_scenario():
    print("=== US-10: Ganymede Launch Window Search & LTS (Refactored) ===")
    
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    from universe.transfer import Transfer
    
    # 1. Initialize Transfer
    transfer = Transfer(origin='ganymede', target='callisto', planner=jax_planner)
    
    # 2. Mission Context
    # Optimal Window Found by optimization_scan.py
    # Dep: 2025-08-18, TOF: 5.52d -> Total DV ~2.4 km/s
    mission_start_iso = "2025-08-18T00:00:00Z"
    flight_time_days = 5.52
    scan_window_days = 2.0 # Narrow scan around optimal
    
    # 3. Find Window
    t_launch_iso = transfer.find_window(
        start_time_iso=mission_start_iso,
        window_days=scan_window_days,
        flight_time_days=flight_time_days
    )
    
    # Calculate Wait Time for Display
    start_dt = datetime.fromisoformat(mission_start_iso.replace('Z', '+00:00'))
    launch_dt = datetime.fromisoformat(t_launch_iso.replace('Z', '+00:00'))
    wait_sec = (launch_dt - start_dt).total_seconds()
    print(f"Parking Duration: {wait_sec/86400:.2f} days")
    
    # 4. Setup Departure (Parking Orbit)
    transfer.setup_departure(
        parking_orbit={'altitude': 500.0, 'body': 'ganymede'}
    )
    
    # 5. Execute Departure
    _, t_burn, dv_mag = transfer.execute_departure(
        thrust=2000.0,
        isp=3000.0,
        initial_mass=1000.0,
        arrival_periapsis_km=500.0 # Target 500km altitude, NOT impact
    )
    
    # 6. Corrections (TCM-1 & TCM-2)
    tcm1_result = transfer.perform_mcc(thrust=2000.0, isp=3000.0, fraction=0.5)
    
    if tcm1_result['final_error_km'] > 10.0:
        print("Executing TCM-2...")
        transfer.perform_mcc(thrust=2000.0, isp=3000.0, fraction=0.9)
    else:
        print("TCM-1 Sufficient. Skipping TCM-2.")
    
    # 7. Propagate Past Arrival (Grazing Analysis)
    print("Propagating past arrival to find Periapsis (High Res)...")
    
    # Get last state
    last_log = transfer.logs[-1][-1]
    last_t_obj = datetime.fromisoformat(last_log['time'].replace('Z', '+00:00'))
    
    # Determine exact arrival time
    t_launch_ref = datetime.fromisoformat(transfer.t_launch_iso.replace('Z', '+00:00'))
    t_arrival_nom = t_launch_ref + timedelta(days=transfer.flight_time_days)
    
    dt_to_arr = (t_arrival_nom - last_t_obj).total_seconds()
    print(f"  Time to Nominal Arrival: {dt_to_arr/3600:.1f} h")
    
    # Propagate: From [Last State] to [Arrival + 6 hours]
    # We want high resolution for the flyby (30s steps)
    dt_coast_total = dt_to_arr + 6 * 3600.0 # +6 hours past arrival
    n_steps_enc = int(dt_coast_total / 30.0) # 30s resolution
    if n_steps_enc < 200: n_steps_enc = 200
    if n_steps_enc > 3000: n_steps_enc = 3000 # Cap for JSON size
    
    print(f"  Propagating {dt_coast_total/3600:.1f} h with {n_steps_enc} steps...")
    
    final_coast = jax_planner.evaluate_trajectory(
        r_start=last_log['position'], v_start=last_log['velocity'],
        t_start_iso=last_log['time'], dt_seconds=dt_coast_total,
        mass=last_log['mass'], n_steps=n_steps_enc
    )
    transfer.add_log(final_coast)
    
    # Analyze
    full_log = []
    for log in transfer.logs:
        full_log.extend(log)
        
    min_dist = float('inf')
    v_rel_at_min = None
    best_time = None
    
    for state in full_log:
        t_iso = state['time']
        p_cal, v_cal = engine.get_body_state('callisto', t_iso)
        r_vec = np.array(state['position']) - np.array(p_cal)
        dist = np.linalg.norm(r_vec)
        
        if dist < min_dist:
            min_dist = dist
            v_rel_at_min = np.array(state['velocity']) - np.array(v_cal)
            best_time = t_iso
            
    # Metrics
    print(f"\n=== Flyby / Grazing Analysis ===")
    R_cal = 2410.3
    alt_p = min_dist - R_cal
    print(f"Periapsis Altitude: {alt_p:.1f} km (at {best_time})")
    
    mu_cal = engine.GM['callisto']
    v_flyby = np.linalg.norm(v_rel_at_min)
    v_circ = np.sqrt(mu_cal / min_dist)
    dv_capture = abs(v_flyby - v_circ)
    
    print(f"V_flyby: {v_flyby*1000:.1f} m/s")
    print(f"V_circ:  {v_circ*1000:.1f} m/s")
    print(f"Est. Capture DV: {dv_capture*1000:.1f} m/s")
    print(f"Departure DV: {dv_mag*1000:.1f} m/s")
    print(f"Total DV (Dep+Cap): {(dv_mag + dv_capture)*1000:.1f} m/s")

    # Final Error (for reference)
    err = min_dist - R_cal - 500.0
    print(f"[Result] Deviation from 500km: {err:.1f} km")
    
    # For plotting reference
    p_cal_arr, _ = engine.get_body_state('callisto', best_time)
    p_gan = engine.get_body_state('ganymede', mission_start_iso)[0]
    p_cal = p_cal_arr
    
    # Plotting
    # Need to handle empty parking log for plotting?
    r_full = np.array([p['position'] for p in full_log])
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_full[:,0], r_full[:,1], r_full[:,2], label='Trajectory', color='cyan')
    ax.scatter(p_gan[0], p_gan[1], p_gan[2], color='orange', label='Ganymede', s=50)
    ax.scatter(p_cal[0], p_cal[1], p_cal[2], color='gray', label='Callisto', s=50)
    
    ax.set_title(f"JAX LTS Mission: Wait {wait_sec/86400:.1f}d + Fly (Err {err:.1f}km)")
    plt.savefig("scenario_jax_lts.png")
    print("Saved scenario_jax_lts.png")
    
    # Export for Viewer
    json_path = "trajectory_lts.json"
    print(f"Exporting (MissionManifest format) to {json_path}...")
    
    start_iso = full_log[0]['time'] # First log entry (could be parking start)
    start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
    
    formatted_timeline = []
    bodies_to_export = ["ganymede", "callisto", "jupiter"]
    
    for entry in full_log:
        curr_iso = entry['time']
        curr_dt = datetime.fromisoformat(curr_iso.replace('Z', '+00:00'))
        dt_seconds = (curr_dt - start_dt).total_seconds()
        
        bodies_pos = {}
        for b in bodies_to_export:
            if b == 'jupiter':
                bodies_pos[b] = [0.0, 0.0, 0.0]
            else:
                p_b, _ = engine.get_body_state(b, curr_iso) 
                bodies_pos[b] = list(p_b)
        
        formatted_timeline.append({
            "time": dt_seconds,
            "position": entry['position'],
            "velocity": entry['velocity'],
            "mass": entry['mass'],
            "bodies": bodies_pos
        })
        
    manifest = {
        "meta": {
            "missionName": "JAX LTS: Ganymede to Callisto",
            "startTime": start_iso,
            "endTime": full_log[-1]['time'],
            "bodies": ["ganymede", "callisto", "jupiter"]
        },
        "timeline": formatted_timeline,
        "maneuvers": [] 
    }
    
    # Add Maneuver Info (Burn)
    # Burn starts at t_launch.
    burn_start_sec = (launch_dt - start_dt).total_seconds()
    
    manifest["maneuvers"].append({
        "startTime": burn_start_sec,
        "duration": t_burn,
        "deltaV": [0.0, 0.0, 0.0], 
        "type": "finite"
    })

    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {json_path}")
    
    # 9. Automated Validation
    print("\n=== Automated Validation ===")
    # Ensure project root is in path if needed, though usually safe
    from universe.tools.validator import TrajectoryValidator
    
    validator = TrajectoryValidator(json_path)
    result = validator.run()
    
    if not result:
        print("!!! VALIDATION FAILED - Check errors above !!!")
    else:
        print("Validation Passed. Trajectory is ready for viewing.")


if __name__ == "__main__":
    run_jax_lts_scenario()
