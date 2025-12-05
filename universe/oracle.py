import numpy as np
from scipy.optimize import minimize
from engine import PhysicsEngine
from datetime import datetime, timedelta

class Oracle:
    def __init__(self, engine: PhysicsEngine):
        self.engine = engine

    def get_orbital_normal(self, pos, vel):
        """Calculate unit normal vector of the orbital plane (h = r x v)."""
        h = np.cross(pos, vel)
        return h / np.linalg.norm(h)

    def rotate_vector_3d(self, vec, axis, angle_rad):
        """Rotate vector around an arbitrary axis using Rodrigues' rotation formula."""
        k = axis / np.linalg.norm(axis)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        
        # v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k.v)*(1-cos(theta))
        return vec * c + np.cross(k, vec) * s + k * np.dot(k, vec) * (1 - c)

    def find_l4_state(self, center_body: str, secondary_body: str, time_iso: str):
        """
        Find the state vector (pos, vel) for a stable L4 orbit.
        Optimizes 6-DOF to minimize RMS drift over 90 days.
        Uses full 3D orbital plane rotation.
        """
        print(f"Solving for L4 stability (90-day RMS, 3D): {secondary_body} relative to {center_body} at {time_iso}")
        
        # 1. Initial Guess (Geometric 3D)
        sec_pos, sec_vel = self.engine.get_body_state(secondary_body, time_iso)
        
        # Calculate Orbital Normal
        normal = self.get_orbital_normal(sec_pos, sec_vel)
        
        # L4 is 60 degrees ahead in the orbital plane
        angle = np.pi / 3
        guess_pos = self.rotate_vector_3d(sec_pos, normal, angle)
        guess_vel = self.rotate_vector_3d(sec_vel, normal, angle)
        
        guess_state = np.concatenate([guess_pos, guess_vel])
        
        # 2. Define Optimization Problem
        # Optimize for 90 days stability
        duration = 90 * 24 * 3600
        
        def objective(state_trial):
            # Propagate for full duration
            # Cache step 600s is fine
            # Propagate for full duration
            final_state = self.engine.propagate(state_trial.tolist(), time_iso, duration)
            
            # Checkpoints for RMS: 30d, 60d, 90d
            # Use t_eval to get all points in one propagation
            checkpoints_days = [30, 60, 90]
            t_eval_checkpoints = [d * 24 * 3600 for d in checkpoints_days]
            
            # Propagate once
            # Propagate once
            states = self.engine.propagate(state_trial.tolist(), time_iso, duration, t_eval=t_eval_checkpoints)
            
            total_drift_sq = 0
            
            for i, d_sec in enumerate(t_eval_checkpoints):
                s = states[i]
                drift = self._calculate_drift(np.array(s[:3]), secondary_body, time_iso, d_sec)
                total_drift_sq += drift**2
            
            # Cost = RMS of drifts
            return np.sqrt(total_drift_sq / len(checkpoints_days))

        # Initial Run
        initial_cost = objective(guess_state)
        print(f"Initial Cost (Geometric 3D): {initial_cost:.2f} km")
        
        # Optimize
        # Nelder-Mead
        res = minimize(objective, guess_state, method='Nelder-Mead', tol=1e-3, options={'maxiter': 1000})
        
        optimized_state = res.x
        final_cost = res.fun
        
        opt_pos = optimized_state[:3]
        opt_vel = optimized_state[3:]
        
        print(f"Optimized Cost (RMS Drift): {final_cost:.2f} km")
        print(f"Position Correction: {np.linalg.norm(opt_pos - guess_pos):.2f} km")
        
        return opt_pos, opt_vel

    def _calculate_drift(self, pos_ship_final, secondary_body, start_time_iso, duration):
        # Calculate Target L4 Position at t_end
        from datetime import datetime, timedelta, timezone
        if start_time_iso.endswith('Z'):
            t_str = start_time_iso[:-1] + '+00:00'
        else:
            t_str = start_time_iso
            
        dt_obj = datetime.fromisoformat(t_str)
        t_end_obj = dt_obj + timedelta(seconds=duration)
        t_end_iso = t_end_obj.isoformat().replace('+00:00', 'Z')
        
        sec_pos_end, sec_vel_end = self.engine.get_body_state(secondary_body, t_end_iso)
        
        # Calculate Normal at t_end (Orbital plane might precess slightly, though slow)
        normal = self.get_orbital_normal(sec_pos_end, sec_vel_end)
        
        # Theoretical L4 at t_end (3D)
        target_l4 = self.rotate_vector_3d(sec_pos_end, normal, np.pi/3)
        
        # Distance
        dist = np.linalg.norm(pos_ship_final - target_l4)
        return dist
