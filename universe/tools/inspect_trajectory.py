
import json
import numpy as np
import sys
import os

def inspect_trajectory(json_path):
    print(f"Inspecting {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    timeline = data['timeline']
    timeline.sort(key=lambda x: x['time'])
    
    print(f"Total Data Points: {len(timeline)}")
    if len(timeline) == 0:
        return

    # Metrics
    t_prev = timeline[0]['time']
    r_prev = np.array(timeline[0]['position'])
    v_prev = np.array(timeline[0]['velocity'])
    
    max_dt = 0.0
    min_dt = float('inf')
    max_pos_jump = 0.0
    max_v_diff = 0.0 # Diff between v_state and dx/dt
    
    anomalies = []

    print("\n--- Scanning for Discontinuities ---")
    
    for i in range(1, len(timeline)):
        curr = timeline[i]
        t_curr = curr['time']
        r_curr = np.array(curr['position'])
        v_curr = np.array(curr['velocity'])
        
        dt = t_curr - t_prev
        dr = r_curr - r_prev
        dist = np.linalg.norm(dr)
        
        # Check Time Continuity
        if dt <= 0:
            anomalies.append(f"Index {i}: Time Duplicate or Reversal (dt={dt}s)")
            
        # Check Position Continuity (Speed Check)
        # Avoid div by zero
        if dt > 0:
            speed_apparent = dist / dt
            # If speed is > 100 km/s (Ganymede transfer is ~10 km/s), it's a jump
            if speed_apparent > 100.0:
                anomalies.append(f"Index {i}: Position Jump! Apparent V={speed_apparent:.1f} km/s (dt={dt:.1f}s, dist={dist:.1f}km)")
                
            # Check Velocity Consistency
            v_mid = (v_prev + v_curr) / 2.0
            v_apparent_vec = dr / dt
            v_error = np.linalg.norm(v_apparent_vec - v_mid)
            
            if v_error > 1.0: # 1 km/s mismatch
                 anomalies.append(f"Index {i}: Kinematic Mismatch! |dr/dt - v_state| = {v_error:.3f} km/s")

        t_prev = t_curr
        r_prev = r_curr
        v_prev = v_curr
        
    if anomalies:
        print(f"Found {len(anomalies)} anomalies:")
        for a in anomalies[:20]: # Show first 20
            print("  " + a)
        if len(anomalies) > 20:
            print("  ... and more.")
    else:
        print("No discontinuities found.")
        
    # Check Gap Sizes
    dts = [timeline[i]['time'] - timeline[i-1]['time'] for i in range(1, len(timeline))]
    if dts:
        print(f"\nTime Steps: Min={min(dts):.1f}s, Max={max(dts):.1f}s, Median={np.median(dts):.1f}s")
        # Find explicit gaps
        threshold = np.median(dts) * 20.0
        gaps = [(i, d) for i, d in enumerate(dts) if d > threshold]
        if gaps:
            print(f"Large Time Gaps detected (> {threshold:.1f}s):")
            for i, d in gaps:
                print(f"  Index {i} -> {i+1}: Gap {d:.1f}s ({d/3600:.1f}h) at t={timeline[i]['time']:.1f}")

    print("\n--- Callisto Approach Analysis ---")
    
    # Constants
    R_CAL = 2410.3
    
    # Find Callisto position in "bodies" (it's [x,y,z])
    # Extract relative distance
    
    close_approach_idx = -1
    min_dist_cal = float('inf')
    
    for i, p in enumerate(timeline):
        if 'bodies' in p and 'callisto' in p['bodies']:
            r_ship = np.array(p['position'])
            r_cal = np.array(p['bodies']['callisto'])
            dist = np.linalg.norm(r_ship - r_cal)
            
            if dist < min_dist_cal:
                min_dist_cal = dist
                close_approach_idx = i
                
    print(f"Minimum Distance to Callisto: {min_dist_cal:.1f} km (at index {close_approach_idx})")
    print(f"Surface Altitude: {min_dist_cal - R_CAL:.1f} km")
    
    if min_dist_cal < R_CAL:
        print("  CRITICAL: Trajectory penetrates surface! (Singularity Risk)")

    # Inspect the approach window
    if close_approach_idx != -1:
        start_win = max(0, close_approach_idx - 20)
        end_win = min(len(timeline), close_approach_idx + 20)
        
        print("\n--- Approach Window Data ---")
        for i in range(start_win, end_win):
            p = timeline[i]
            r_ship = np.array(p['position'])
            r_cal = np.array(p['bodies']['callisto'])
            dr = r_ship - r_cal
            dist = np.linalg.norm(dr)
            alt = dist - R_CAL
            
            dt_step = 0.0
            v_rel = 0.0
            if i > 0:
                dt_step = p['time'] - timeline[i-1]['time']
                if dt_step > 0.001:
                    # Relative Velocity
                    r_cal_prev = np.array(timeline[i-1]['bodies']['callisto'])
                    r_ship_prev = np.array(timeline[i-1]['position'])
                    dr_prev = r_ship_prev - r_cal_prev
                    v_rel = np.linalg.norm(dr - dr_prev) / dt_step
            
            print(f"[{i}] t={p['time']:.1f}, Alt={alt:.1f}km, Dist={dist:.1f}km, dt={dt_step:.1f}s, V_rel={v_rel:.1f} km/s")

if __name__ == "__main__":
    inspect_trajectory("trajectory_lts.json")
