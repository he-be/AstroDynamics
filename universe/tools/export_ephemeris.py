import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from universe.engine import PhysicsEngine

def generate_ephemeris(output_file, start_date="2025-01-01T00:00:00Z", days=365, step_hours=4):
    """
    Generates a MissionManifest strictly for Ephemeris (Planetary Positions).
    Spacecraft position is dummy.
    """
    print(f"Generating Ephemeris for {days} days from {start_date}...")
    
    engine = PhysicsEngine()
    # engine.load_kernels() # Kernels loaded in __init__
    
    # Time Setup
    t0 = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    t_end = t0 + timedelta(days=days)
    
    bodies = ['jupiter', 'io', 'europa', 'ganymede', 'callisto']
    timeline = []
    
    current_time = t0
    step_delta = timedelta(hours=step_hours)
    
    # Epoch (Unix Timestamp of Start)
    epoch_ts = t0.timestamp()
    
    count = 0
    while current_time <= t_end:
        t_iso = current_time.isoformat().replace('+00:00', 'Z')
        t_rel = current_time.timestamp() - epoch_ts
        
        # Collect Body States
        current_bodies = {}
        for b in bodies:
            if b == 'jupiter': continue # 0,0,0
            try:
                p, _ = engine.get_body_state(b, t_iso)
                current_bodies[b] = list(p)
            except Exception as e:
                pass
        
        timeline.append({
            'time': t_rel,
            'position': [0, 0, 0], # Dummy Spacecraft
            'velocity': [0, 0, 0],
            'mass': 1000.0,
            'bodies': current_bodies
        })
        
        current_time += step_delta
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} points...")

    # Manifest
    manifest = {
        'meta': {
            'missionName': 'Solar System Ephemeris',
            'startTime': start_date,
            'endTime': t_end.isoformat().replace('+00:00', 'Z'),
            'bodies': bodies
        },
        'timeline': timeline,
        'maneuvers': []
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=0) # Compact
        
    print(f"Saved ephemeris to {output_file} ({len(timeline)} points)")

if __name__ == "__main__":
    output_path = "web-viewer/public/ephemeris_default.json"
    generate_ephemeris(output_path)
