
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
    
    # 7. Collect Data for Plotting
    full_log = []
    for log in transfer.logs:
        full_log.extend(log)
        
    last_state = full_log[-1]
    p_cal_arr, _ = engine.get_body_state('callisto', last_state['time'])
    err = np.linalg.norm(np.array(last_state['position']) - np.array(p_cal_arr))
    print(f"[Result] Final Error: {err:.1f} km")
    
    center_body_pos, _ = engine.get_body_state('ganymede', mission_start_iso) # Just for ref
    p_gan = center_body_pos
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
