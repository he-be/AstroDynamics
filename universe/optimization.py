import numpy as np
from datetime import datetime, timedelta
from planning import solve_lambert

class PorkchopOptimizer:
    def __init__(self, engine):
        self.engine = engine
        self.mu_primary = engine.GM['jupiter']
        self.mu_bodies = engine.GM 
        # e.g. self.mu_bodies['ganymede']
    
    def calculate_delta_v(self, t_dep_iso, dt_sec, body_dep, body_arr, r_park_dep, r_park_arr):
        """
        Calculate Total Delta V (Departure + Arrival) for a transfer.
        Uses Hyperbolic Excess Velocity logic (Oberth effect).
        """
        # 1. State Vectors of Moons
        t_arr_obj = datetime.fromisoformat(t_dep_iso.replace('Z', '+00:00')) + timedelta(seconds=dt_sec)
        t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
        
        p_dep, v_dep_body = self.engine.get_body_state(body_dep, t_dep_iso)
        p_arr, v_arr_body = self.engine.get_body_state(body_arr, t_arr_iso)
        
        # 2. Lambert Solution
        try:
            v_dep_vec, v_arr_vec = solve_lambert(np.array(p_dep), np.array(p_arr), dt_sec, self.mu_primary)
        except Exception:
            return float('inf'), None, None
            
        # 3. Departure Delta V (from Parking Orbit)
        v_inf_dep_vec = v_dep_vec - np.array(v_dep_body)
        v_inf_dep_sq = np.dot(v_inf_dep_vec, v_inf_dep_vec)
        
        mu_dep = self.mu_bodies.get(body_dep, 0)
        v_circ_dep = np.sqrt(mu_dep / r_park_dep)
        if mu_dep > 0:
            # Oberth: Delta V = sqrt(V_inf^2 + 2*mu/r) - sqrt(mu/r)
            # Actually V_per = sqrt(V_inf^2 + 2*mu/r).
            # Delta V = V_per - V_circ.
            dv_dep = np.sqrt(v_inf_dep_sq + 2*mu_dep/r_park_dep) - v_circ_dep
        else:
            # No gravity well (e.g. L-point departure?)
            dv_dep = np.sqrt(v_inf_dep_sq)
            
        # 4. Arrival Delta V (Capture into Parking Orbit)
        v_inf_arr_vec = v_arr_vec - np.array(v_arr_body)
        v_inf_arr_sq = np.dot(v_inf_arr_vec, v_inf_arr_vec)
        
        mu_arr = self.mu_bodies.get(body_arr, 0)
        v_circ_arr = np.sqrt(mu_arr / r_park_arr)
        if mu_arr > 0:
            dv_arr = np.sqrt(v_inf_arr_sq + 2*mu_arr/r_park_arr) - v_circ_arr
        else:
            dv_arr = np.sqrt(v_inf_arr_sq)
            
        total_dv = dv_dep + dv_arr
        return total_dv, t_dep_iso, dt_sec

    def optimize_window(self, body_dep, body_arr, 
                        t_start_iso, window_duration_days, 
                        flight_time_range_days, 
                        r_park_dep, r_park_arr,
                        step_days=1.0, dt_step_days=0.5):
        """
        Simple Grid Search for optimal transfer.
        """
        start_dt = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
        
        best_dv = float('inf')
        best_params = None # (t_dep, dt)
        
        # Grid Setup
        dep_steps = int(window_duration_days / step_days)
        ft_min, ft_max = flight_time_range_days
        dt_steps = int((ft_max - ft_min) / dt_step_days)
        
        print(f"Optimizing {body_dep}->{body_arr}: {dep_steps}x{dt_steps} grid...")
        
        for i in range(dep_steps):
            t_current = start_dt + timedelta(days=i * step_days)
            t_iso = t_current.isoformat().replace('+00:00', 'Z')
            
            for j in range(dt_steps):
                dt_days = ft_min + j * dt_step_days
                dt_sec = dt_days * 86400.0
                
                dv, _, _ = self.calculate_delta_v(t_iso, dt_sec, body_dep, body_arr, r_park_dep, r_park_arr)
                
                if dv < best_dv:
                    best_dv = dv
                    best_params = (t_iso, dt_days)
                    
        return best_dv, best_params
