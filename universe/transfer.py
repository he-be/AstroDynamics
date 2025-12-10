
import numpy as np
from datetime import datetime, timedelta
from universe.jax_planning import JAXPlanner, MU_JUP
from universe.planning import solve_lambert

class Transfer:
    def __init__(self, origin: str, target: str, planner: JAXPlanner):
        self.origin = origin
        self.target = target
        self.planner = planner
        self.engine = planner.engine
        
        # State
        self.logs = [] # List of logs (parking, burn, coast, mcc...)
        self.events = [] # List of event dicts
        self.t_launch_iso = None
        self.flight_time_days = None
        self.launch_state = None # State at execution start
        
    def find_window(self, start_time_iso: str, window_days: float, flight_time_days: float, dt_step_hours: float = 6.0):
        """
        Finds optimal launch window.
        """
        print(f"Scanning {window_days} days for optimal window ({self.origin} -> {self.target})...")
        opt_launch_dt, opt_v_dep = self.planner.find_optimal_launch_window(
            t_window_start_iso=start_time_iso,
            window_duration_days=window_days,
            flight_time_days=flight_time_days,
            dt_step_hours=dt_step_hours
        )
        # Store result
        t_iso = opt_launch_dt.isoformat().replace('+00:00', 'Z')
        if not t_iso.endswith('Z'): t_iso += 'Z'
        
        self.t_launch_iso = t_iso
        self.flight_time_days = flight_time_days
        self.opt_v_dep = opt_v_dep # Lambert Velocity Vector
        
        print(f"--> Selected Launch: {self.t_launch_iso}")
        return t_iso
        
    def setup_departure(self, parking_orbit: dict = None, wait: bool = True):
        """
        Simulates parking orbit and optimized phasing.
        parking_orbit: { 'altitude': 500.0, 'body': 'ganymede' }
        """
        if self.t_launch_iso is None:
            raise ValueError("Call find_window first.")
            
        # 1. Determine Window Start vs Launch Time (Parking Duration)
        # Assumes caller knows "Mission Start". If not provided, assume window start was mission start?
        # Let's verify usage. In scenario: mission_start != launch_time.
        # We need mission_start passed in? 
        # Or we just assume we start at "some time" and wait until t_launch_iso?
        # Let's say we park FROM t_launch_iso - duration? No.
        # Usually we start at T0 and wait until Launch.
        pass 
        
        # For simplification, `setup_departure` assumes we are AT `t_launch_iso` roughly.
        # But Phase Matching adjusts `t_launch_iso` slightly.
        
        if parking_orbit:
            print("Simulating Parking Orbit & Phasing...")
            body = parking_orbit.get('body', self.origin)
            alt = parking_orbit.get('altitude', 500.0)
            
            # Simple Circular Setup
            p_body, v_body = self.engine.get_body_state(body, self.t_launch_iso)
            mu_body = self.engine.GM[body]
            R_body = 2634.1 # Ganymede (TODO: Get from constants/engine)
            if body == 'ganymede': R_body = 2634.1
            
            r_mag = R_body + alt
            v_circ = np.sqrt(mu_body / r_mag)
            
            # State in Jovicentric frame
            # Planar approximation
            r_park = np.array(p_body) + np.array([r_mag, 0.0, 0.0])
            v_park = np.array(v_body) + np.array([0.0, v_circ, 0.0])
            
            # Calculate Period
            period = 2 * np.pi * np.sqrt(r_mag**3 / mu_body)
            print(f"  Orbit Period: {period/60.0:.1f} mins")
            
            # Optimize Phase
            best_step, scan_log, best_dv = self.planner.find_optimal_parking_phase(
                r_parking=list(r_park),
                v_parking=list(v_park),
                t_parking_iso=self.t_launch_iso, # This is technically "nominal" launch
                period_seconds=period * 1.5, # Scan 1.5 orbits around nominal
                flight_time_seconds=self.flight_time_days * 86400.0,
                target_body=self.target
            )
            
            self.logs.append(scan_log)
            
            # Update Launch State
            self.launch_state = best_step
            self.t_launch_iso = best_step['time'] # Updated
            print(f"  Phased Launch Time: {self.t_launch_iso}")
            
        else:
            # Direct Launch state
            pass
            
    def execute_departure(self, thrust: float, isp: float, initial_mass: float):
        """
        Plans and executes the departure burn (Finite Burn).
        """
        if self.launch_state is None:
             raise ValueError("Setup departure first.")
             
        # 1. Refine Target
        r_launch = np.array(self.launch_state['position'])
        v_launch = np.array(self.launch_state['velocity']) # Parking V
        
        t_launch_obj = datetime.fromisoformat(self.t_launch_iso.replace('Z', '+00:00'))
        t_arr_obj = t_launch_obj + timedelta(days=self.flight_time_days)
        t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
        
        p_target, _ = self.engine.get_body_state(self.target, t_arr_iso)
        dt_sec = self.flight_time_days * 86400.0
        
        # 2. Oberth / Lambert Guess
        # Calculate Lambert Arc (Interplanetary/Jovicentric)
        try:
             v_lamb, _ = solve_lambert(r_launch, np.array(p_target), dt_sec, MU_JUP)
             
             # Refine Guess using Oberth Effect (compensate for local gravity)
             # IF we are departing a moon (parking orbit context)
             # self.origin is 'ganymede'
             if self.origin in self.engine.GM: 
                 p_body, v_body = self.engine.get_body_state(self.origin, self.t_launch_iso)
                 mu_body = self.engine.GM[self.origin]
                 
                 # V_inf relative to body
                 v_inf_vec = v_lamb - np.array(v_body)
                 v_inf_sq = np.linalg.norm(v_inf_vec)**2
                 
                 # Escape Velocity from local position
                 r_rel = np.linalg.norm(r_launch - np.array(p_body))
                 v_esc_local_sq = 2 * mu_body / r_rel
                 
                 # Required Velocity magnitude at Periapsis
                 v_needed_mag = np.sqrt(v_inf_sq + v_esc_local_sq)
                 
                 # Direction: Approximation using V_inf direction
                 u_inf = v_inf_vec / np.linalg.norm(v_inf_vec)
                 v_guess = np.array(v_body) + v_needed_mag * u_inf
                 print("  Using Oberth-Corrected Guess.")
             else:
                 v_guess = v_lamb

        except Exception as e:
             print(f"Lambert/Oberth failed: {e}")
             v_guess = (np.array(p_target) - r_launch) / dt_sec
             
        v_impulse_refined = self.planner.solve_impulsive_shooting(
            r_start=list(r_launch),
            t_start_iso=self.t_launch_iso,
            dt_seconds=dt_sec,
            r_target=list(p_target),
            initial_v_guess=list(v_guess)
        )
        
        dv_impulse = np.array(v_impulse_refined) - v_launch
        dv_mag = np.linalg.norm(dv_impulse)
        print(f"Departure Delta-V: {dv_mag*1000:.1f} m/s")
        
        # 3. Finite Burn
        g0 = 9.80665
        ve = isp * g0 / 1000.0
        m_dot = thrust / (ve * 1000.0)
        t_burn = (initial_mass * (1.0 - np.exp(-dv_mag/ve))) / m_dot
        t_burn *= 1.01
        
        # Heuristic Offset
        t_start_burn_obj = t_launch_obj - timedelta(seconds=t_burn/2.0)
        t_start_burn_iso = t_start_burn_obj.isoformat().replace('+00:00', 'Z')
        
        # Propagate parking to start burn (Back or Fwd)
        p_state_time_obj = datetime.fromisoformat(self.launch_state['time'].replace('Z', '+00:00'))
        dt_prop = (t_start_burn_obj - p_state_time_obj).total_seconds()
        
        # Use existing trajectory or propagate
        # Assuming launch_state was from a log, we might need new propagation if dt < 0
        # If dt > 0, we propagate.
        pre_burn_log = self.planner.evaluate_trajectory(
            r_start=self.launch_state['position'], v_start=self.launch_state['velocity'],
            t_start_iso=self.launch_state['time'], dt_seconds=dt_prop,
            mass=initial_mass, n_steps=50
        )
        start_state = pre_burn_log[-1]
        
        # Solve LTS
        print("Solving Departure Burn...")
        dt_coast = dt_sec - t_burn/2.0
        params = self.planner.solve_finite_burn_coast(
            r_start=start_state['position'], v_start=start_state['velocity'],
            t_start_iso=start_state['time'],
            t_burn_seconds=t_burn, t_coast_seconds=dt_coast,
            target_pos=list(p_target),
            mass_init=initial_mass,
            thrust=thrust, isp=isp,
            impulse_vector=dv_impulse,
            tol_km=100.0
        )
        
        # Execute
        burn_log = self.planner.evaluate_burn(
             r_start=start_state['position'], v_start=start_state['velocity'],
             t_start_iso=start_state['time'], dt_seconds=t_burn,
             lts_params=params, thrust=thrust, isp=isp, mass_init=initial_mass
        )
        self.logs.append(burn_log)
        self.burn_end_state = burn_log[-1]
        
        return burn_log, t_burn, dv_mag
        
    def perform_mcc(self, thrust: float, isp: float, fraction: float = 0.5):
        """
        Executes MCC at fraction of flight time.
        """
        end_burn = self.burn_end_state
        t_launch_obj = datetime.fromisoformat(self.t_launch_iso.replace('Z', '+00:00'))
        dt_flight = self.flight_time_days * 86400.0
        
        t_mcc_obj = t_launch_obj + timedelta(seconds=dt_flight * fraction)
        t_mcc_iso = t_mcc_obj.isoformat().replace('+00:00', 'Z')
        t_burn_end_obj = datetime.fromisoformat(end_burn['time'].replace('Z', '+00:00'))
        
        dt_coast = (t_mcc_obj - t_burn_end_obj).total_seconds()
        print(f"Coasting to MCC: {dt_coast/3600:.1f} h")
        
        coast_log = self.planner.evaluate_trajectory(
            r_start=end_burn['position'], v_start=end_burn['velocity'],
            t_start_iso=end_burn['time'], dt_seconds=dt_coast,
            mass=end_burn['mass'], n_steps=100
        )
        self.logs.append(coast_log)
        
        # Plan MCC
        t_arr_obj = t_launch_obj + timedelta(seconds=dt_flight)
        t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
        p_target_arr, _ = self.engine.get_body_state(self.target, t_arr_iso)
        
        result = self.planner.plan_correction_maneuver(
            current_state=coast_log[-1],
            target_pos=list(p_target_arr),
            target_time_iso=t_arr_iso,
            thrust=thrust, isp=isp,
            tolerance_km=10.0,
            previous_state=end_burn
        )
        
        self.logs.append(result['maneuver_log'])
        self.last_result = result
        
        return result
        
