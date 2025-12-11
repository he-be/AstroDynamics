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

    def validate_kinematics(self, v_tol=1.0):
        """
        Checks if the position change matches the stored velocity.
        v_tol: Tolerance in km/s (Defaul 1.0 km/s due to integration/sampling diffs)
        """
        for i in range(1, len(self.timeline)):
            prev = self.timeline[i-1]
            curr = self.timeline[i]
            
            dt = curr['time'] - prev['time']
            if dt <= 0.001: continue
            
            r_prev = np.array(prev['position'])
            r_curr = np.array(curr['position'])
            
            v_prev = np.array(prev['velocity'])
            v_curr = np.array(curr['velocity'])
            
            # Average velocity state
            v_state_avg = (v_prev + v_curr) / 2.0
            
            # Apparent velocity
            dr = r_curr - r_prev
            v_apparent = dr / dt
            
            # Error
            v_diff = np.linalg.norm(v_apparent - v_state_avg)
            
            # Speed check (Teleportation)
            speed_apparent = np.linalg.norm(v_apparent)
            if speed_apparent > 100.0: 
                 self.errors.append(f"Index {i}: Teleportation? Apparent V={speed_apparent:.1f} km/s (State V={np.linalg.norm(v_state_avg):.1f})")

            if v_diff > v_tol:
                msg = f"Index {i}: Kinematic Mismatch |dr/dt - v| = {v_diff:.3f} km/s (dt={dt:.1f}s)"
                self.warnings.append(msg)
                
                # Elevate to error if severe
                if v_diff > 5.0:
                    details = f"  Pos Jump: {np.linalg.norm(dr):.1f} km. Apparent V: {speed_apparent:.1f}. State V: {np.linalg.norm(v_state_avg):.1f}"
                    self.errors.append(f"Index {i}: SEVERE Kinematic Mismatch ({v_diff:.1f} km/s)\n    {details}")

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
        self.validate_kinematics()
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
