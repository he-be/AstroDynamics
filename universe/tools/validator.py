import json
import numpy as np
import sys
import os

class TrajectoryValidator:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = None
        self.timeline = []
        self.errors = []
        self.warnings = []
        
    def load(self):
        try:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
            self.timeline = self.data.get('timeline', [])
            self.timeline.sort(key=lambda x: x['time'])
            return True
        except Exception as e:
            self.errors.append(f"Failed to load JSON: {str(e)}")
            return False

    def validate_structure(self):
        if not self.timeline:
            self.errors.append("Timeline is empty.")
            return

        required_keys = ['time', 'position', 'velocity', 'mass']
        for i, entry in enumerate(self.timeline):
            for key in required_keys:
                if key not in entry:
                    self.errors.append(f"Index {i}: Missing key '{key}'")
                    return # Stop early to avoid spam

    def validate_time(self):
        if len(self.timeline) < 2:
            return

        t_prev = self.timeline[0]['time']
        dts = []
        
        for i in range(1, len(self.timeline)):
            t_curr = self.timeline[i]['time']
            dt = t_curr - t_prev
            
            if dt <= 0:
                self.errors.append(f"Index {i}: Time Duplicate or Reversal (dt={dt}s)")
            
            dts.append(dt)
            t_prev = t_curr
            
        median_dt = np.median(dts)
        for i, dt in enumerate(dts):
             if dt > 20 * median_dt and dt > 600.0: # 10 minutes
                 self.warnings.append(f"Index {i}->{i+1}: Large Time Gap ({dt:.1f}s)")

    def validate_kinematics_and_maneuvers(self, v_tol=1.0):
        """
        Checks kinematics, reconciling velocity jumps with known maneuvers.
        Handles finite burns by verifying acceleration magnitude.
        """
        maneuvers = self.data.get('maneuvers', [])
        from datetime import datetime
        
        # Sort maneuvers
        maneuver_map = []
        if maneuvers and 'meta' in self.data:
            meta_start = self.data['meta']['startTime']
            t0_obj = datetime.fromisoformat(meta_start.replace('Z', '+00:00'))
            
            for m in maneuvers:
                 if isinstance(m['time'], str):
                     m_dt = datetime.fromisoformat(m['time'].replace('Z', '+00:00'))
                     t_sec = (m_dt - t0_obj).total_seconds()
                 else:
                     t_sec = float(m['time'])
                     
                 maneuver_map.append({
                     't_start': t_sec,
                     't_end': t_sec + m.get('duration', 0.0),
                     'dv': m.get('delta_v', 0.0), # Total planned DV
                     'type': m.get('type', 'unknown'),
                     'duration': m.get('duration', 0.0),
                     'obs_dv': 0.0,
                     'matched': False
                 })
            # print(f"[Validator] Loaded {len(maneuver_map)} expected maneuvers.")

        for i in range(1, len(self.timeline)):
            prev = self.timeline[i-1]
            curr = self.timeline[i]
            
            dt = curr['time'] - prev['time']
            if dt <= 0.001: continue
            
            r_prev = np.array(prev['position'])
            r_curr = np.array(curr['position'])
            
            v_prev = np.array(prev['velocity'])
            v_curr = np.array(curr['velocity'])
            
            v_state_avg = (v_prev + v_curr) / 2.0
            dr = r_curr - r_prev
            v_apparent = dr / dt
            v_diff = np.linalg.norm(v_apparent - v_state_avg)
            
            dv_step = np.linalg.norm(v_curr - v_prev)

            # Check if inside a maneuver
            t_mid = (curr['time'] + prev['time']) / 2.0
            active_maneuver = None
            
            for m in maneuver_map:
                # Tolerance: start-10s to end+10s
                if (m['t_start'] - 10.0) <= t_mid <= (m['t_end'] + 10.0):
                    active_maneuver = m
                    break
            
            if active_maneuver:
                # Finite Burn Check
                # Expected DV rate = Total DV / Duration
                # But mass changes, so acc increases. Linear approx: avg acc.
                if active_maneuver['duration'] > 1.0:
                    avg_acc = active_maneuver['dv'] / active_maneuver['duration']
                    exp_dv_step = avg_acc * dt
                    # Allow 50% variance (mass effect + geometry) + 0.1 km/s noise
                    tolerance = max(0.1, exp_dv_step * 0.5)
                else:
                    # Impulsive
                    exp_dv_step = active_maneuver['dv']
                    tolerance = max(1.0, active_maneuver['dv'] * 0.2)
                
                # Check agreement
                if abs(dv_step - exp_dv_step) < tolerance:
                    # Satisfied
                    active_maneuver['obs_dv'] += dv_step
                    active_maneuver['matched'] = True 
                    continue
                else:
                    # Mismatch in rate
                    # Maybe it's just the edge of the burn?
                    if dv_step < tolerance: # Negligible
                        continue
                    # Else warning
                    self.warnings.append(f"Index {i}: Burn Rate Mismatch at t={t_mid:.1f}. Obs {dv_step:.3f} vs Exp {exp_dv_step:.3f}")
            
            # If not a verified burn step, check Physics
            if v_diff > v_tol:
                msg = f"Index {i}: Kinematic Mismatch |dr/dt - v| = {v_diff:.3f} km/s (dt={dt:.1f}s)"
                self.warnings.append(msg)
                
                if v_diff > 5.0:
                     self.errors.append(f"Index {i}: SEVERE Kinematic Mismatch ({v_diff:.1f} km/s) [No matching maneuver]")

        # Final Verification of Maneuvers
        for m in maneuver_map:
            if not m['matched']:
                self.warnings.append(f"Missing Maneuver: {m['type']} (Exp {m['dv']:.2f} km/s)")
            else:
                # Check Total DV
                diff = abs(m['obs_dv'] - m['dv'])
                # If error > 20%
                if diff > max(0.2, m['dv'] * 0.2):
                     self.warnings.append(f"Maneuver DV Mismatch ({m['type']}): Obs {m['obs_dv']:.2f} vs Exp {m['dv']:.2f} km/s")
                else:
                     pass # print(f"Verified {m['type']}: {m['obs_dv']:.2f}/{m['dv']:.2f} km/s")

    def validate_safety(self):
        # Body Radii
        radii = {
            'callisto': 2410.3,
            'ganymede': 2634.1,
            'europa': 1560.8,
            'io': 1821.6,
            'jupiter': 71492.0
        }
        
        for i, entry in enumerate(self.timeline):
            if 'bodies' not in entry: continue
            
            r_ship = np.array(entry['position'])
            
            for body, pos in entry['bodies'].items():
                if body in radii:
                    r_body = np.array(pos)
                    dist = np.linalg.norm(r_ship - r_body)
                    limit = radii[body]
                    
                    if dist < limit:
                         self.errors.append(f"Index {i}: Surface Collision with {body}! Alt={dist-limit:.1f} km")
                    elif dist < limit + 100.0:
                         self.warnings.append(f"Index {i}: Grazing {body} (Alt={dist-limit:.1f} km)")

    def run(self):
        print(f"[Validator] Inspecting {self.json_path}...")
        if not self.load():
            return False
            
        self.validate_structure()
        self.validate_time()
        self.validate_kinematics_and_maneuvers()
        self.validate_safety()
        
        # Report
        if self.warnings:
            print(f"[Validator] Found {len(self.warnings)} Warnings:")
            for w in self.warnings[:5]: print("  " + w)
            if len(self.warnings) > 5: print(f"  ... and {len(self.warnings)-5} more.")
            
        if self.errors:
            print(f"[Validator] CRITICAL: Found {len(self.errors)} Errors!")
            for e in self.errors[:10]: print("  " + e)
            return False
        
        print("[Validator] Trajectory is VALID.")
        return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validator.py <json_path>")
        sys.exit(1)
        
    v = TrajectoryValidator(sys.argv[1])
    if not v.run():
        sys.exit(1)
