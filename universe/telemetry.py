import json
from datetime import datetime
import numpy as np

def export_mission_manifest(controller, filename, mission_name="Mission", bodies=None):
    """
    Exports flight controller history to MissionManifest JSON format.
    
    Args:
        controller: FlightController instance
        filename: Output filename (e.g. 'mission.json')
        mission_name: Name of the mission
        bodies: List of relevant bodies (e.g. ['jupiter', 'ganymede'])
    """
    if bodies is None:
        bodies = ['jupiter', 'ganymede']

    log = controller._trajectory_log
    if not log:
        print("Warning: No trajectory data to export.")
        return

    # Determine Epoch (Start Time)
    start_iso = log[0]['time']
    end_iso = log[-1]['time']
    
    # Helper: ISO to Seconds from Epoch
    def parse_time(iso_str):
        if iso_str.endswith('Z'): iso_str = iso_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()

    epoch_ts = parse_time(start_iso)

    # 1. Timeline
    timeline = []
    for entry in log:
        t_ts = parse_time(entry['time'])
        t_rel = t_ts - epoch_ts
        
        state = entry['state'] # [rx,ry,rz, vx,vy,vz]
        mass = entry['mass']
        
        timeline.append({
            'time': t_rel,
            'position': list(state[:3]),
            'velocity': list(state[3:6]),
            'mass': float(mass)
        })

    # 2. Maneuvers
    maneuvers = []
    for m in controller.maneuver_log:
        t_ts = parse_time(m['time_iso'])
        t_rel = t_ts - epoch_ts
        
        # Determine Delta V Vector
        # Some old logs might not have 'delta_v_vec_km_s', fallback to generic
        dv_vec = m.get('delta_v_vec_km_s', [0,0,0])
        
        maneuvers.append({
            'startTime': t_rel,
            'duration': float(m['duration_s']),
            'deltaV': dv_vec,
            'type': m.get('type', 'finite')
        })

    # 3. Construct Manifest
    manifest = {
        'meta': {
            'missionName': mission_name,
            'startTime': start_iso,
            'endTime': end_iso,
            'bodies': bodies
        },
        'timeline': timeline,
        'maneuvers': maneuvers
    }

    # Write File
    with open(filename, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[Telemetry] Exported mission manifest to {filename} ({len(timeline)} points)")
