
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
    mission_start_iso = "2025-07-29T12:00:00Z"
    flight_time_days = 4.0
    scan_window_days = 20.0
    
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
        initial_mass=1000.0
    )
    
    # 6. Corrections (TCM-1 & TCM-2)
    tcm1_result = transfer.perform_mcc(thrust=2000.0, isp=3000.0, fraction=0.5)
    
    if tcm1_result['final_error_km'] > 10.0:
        print("Executing TCM-2...")
        transfer.perform_mcc(thrust=2000.0, isp=3000.0, fraction=0.9)
    else:
        print("TCM-1 Sufficient. Skipping TCM-2.")
    
    # 7. Propagate Past Arrival (Grazing Analysis)
    print("Propagating past arrival to find Periapsis...")
    
    # Get last state
    last_log = transfer.logs[-1][-1]
    
    # Propagate for 1 more day
    extra_coast_sec = 86400.0 
    final_coast = jax_planner.evaluate_trajectory(
        r_start=last_log['position'], v_start=last_log['velocity'],
        t_start_iso=last_log['time'], dt_seconds=extra_coast_sec,
        mass=last_log['mass'], n_steps=200
    )
    transfer.logs.append(final_coast)
    
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


if __name__ == "__main__":
    run_jax_lts_scenario()
