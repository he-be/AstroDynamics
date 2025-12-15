
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
        self.flight_time_days = None
        self.launch_state = None # State at execution start
        self.target_pos_vector = None # Targeted arrival position (B-Plane)
        
    def add_log(self, new_log: list):
        """
        Append a new log segment, handling overlaps by truncating history (Branching).
        Enforces strict time monotonicity.
        """
        if not new_log: return
        
        t_new_start = datetime.fromisoformat(new_log[0]['time'].replace('Z', '+00:00'))
        
        # Backtrack: Remove future history if new log starts in the past
        while self.logs:
            last_segment = self.logs[-1]
            if not last_segment:
                self.logs.pop()
                continue
                
            t_last_end = datetime.fromisoformat(last_segment[-1]['time'].replace('Z', '+00:00'))
            
            # If new start is AFTER last end (with tolerance), we are good
            if (t_new_start - t_last_end).total_seconds() >= -0.001:
                break
            
            # If Overlap: Truncate the last segment
            # Find cut point
            # We want to keep points strictly BEFORE t_new_start
            cut_idx = -1
            for i in range(len(last_segment)):
                t_p = datetime.fromisoformat(last_segment[i]['time'].replace('Z', '+00:00'))
                if (t_p - t_new_start).total_seconds() >= -0.001:
                    # Found point >= new_start. Cut everything from here onwards.
                    cut_idx = i
                    break
            
            if cut_idx != -1:
                print(f"  [Transfer] Overlap detected. Truncating log segment at index {cut_idx} (Time {last_segment[cut_idx]['time']})")
                self.logs[-1] = last_segment[:cut_idx]
                # If segment became empty, pop it
                if not self.logs[-1]:
                    self.logs.pop()
            else:
                 # If we are here, it means the entire segment is BEFORE new_start?
                 # But outer loop check said t_new_start < t_last_end
                 # This implies t_new_start is INSIDE the segment (found by loop)
                 # OR t_new_start is BEFORE the segment start?
                 t_last_start = datetime.fromisoformat(last_segment[0]['time'].replace('Z', '+00:00'))
                 if t_new_start < t_last_start:
                     # Entire segment is in future. Drop it.
                     self.logs.pop()
                 else:
                     break
        
        # Check for precise duplicate at usage boundary
        if self.logs and self.logs[-1]:
             t_last_end = datetime.fromisoformat(self.logs[-1][-1]['time'].replace('Z', '+00:00'))
             # Loop to trim leading points that match within tolerance
             while new_log:
                 t_new_start = datetime.fromisoformat(new_log[0]['time'].replace('Z', '+00:00'))
                 delta = (t_new_start - t_last_end).total_seconds()
                 if delta < 0.05:
                     # print(f"[Transfer] Popping duplicate time {t_new_start} (dt={delta:.3f})")
                     new_log.pop(0)
                     # print(f"[Transfer] Popping duplicate time {t_new_start} (dt={delta:.3f})")
                     new_log.pop(0)
                 else:
                     break

        # Append
        if new_log:
             self.logs.append(new_log)

    def find_window(self, start_time_iso: str, window_days: float, flight_time_days: float, dt_step_hours: float = 6.0):
        """
        Finds optimal launch window.
        Uses analytic Hohmann prediction to center the search.
        """
        from universe.planning import predict_hohmann_window_circular
        
        # 1. Analytic Prediction (Hohmann Seeding)
        print(f"Predicting Hohmann Window for {self.origin} -> {self.target}...")
        
        t_start_obj = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        
        # Get States for phase calculation
        p_org, _ = self.engine.get_body_state(self.origin, start_time_iso)
        p_tgt, _ = self.engine.get_body_state(self.target, start_time_iso)
        
        r1 = np.linalg.norm(p_org)
        r2 = np.linalg.norm(p_tgt)
        mu = self.engine.GM['jupiter']
        
        # Phase (Angle of Target relative to Origin)
        # Assuming Z-axis is normal (planar approx)
        theta1 = np.arctan2(p_org[1], p_org[0])
        theta2 = np.arctan2(p_tgt[1], p_tgt[0])
        phase_curr = theta2 - theta1
        
        dt_wait, dt_flight_hohmann, phi_req = predict_hohmann_window_circular(
            r1, r2, phase_curr, mu
        )
        
        print(f"  Analytic Prediction: Wait {dt_wait/86400:.2f}d, Flight {dt_flight_hohmann/86400:.2f}d")
        
        # Center the scan window around predicted date
        predicted_launch = t_start_obj + timedelta(seconds=dt_wait)
        
        # We start scan slightly before prediction to catch local optimum
        scan_start = predicted_launch - timedelta(days=2.0) 
        scan_duration = 5.0 # Narrow scan +/- 2.5 days is sufficient if prediction is good
        
        print(f"  Refining Search Window: {scan_start.isoformat()} (+{scan_duration}d scan)")
        
        # Use Hohmann Flight Time as the guide
        # User input 'flight_time_days' is an upper bound constraint.
        if dt_flight_hohmann > flight_time_days * 86400.0:
             print(f"  [Warning] Hohmann Time ({dt_flight_hohmann/86400:.1f}d) > Constraint ({flight_time_days}d). Efficiency loss expected.")
             scan_flight_time_bound = flight_time_days
        else:
             # If constraint allows, scan around Hohmann time (up to 1.2x just to be safe)
             scan_flight_time_bound = dt_flight_hohmann / 86400.0 * 1.2
             if scan_flight_time_bound > flight_time_days:
                 scan_flight_time_bound = flight_time_days
            
        scan_start_iso = scan_start.isoformat().replace('+00:00', 'Z')
        
        # 2. Detailed Scan
        opt_launch_dt, opt_flight_sec, opt_v_dep = self.planner.find_optimal_launch_window(
            t_window_start_iso=scan_start_iso,
            window_duration_days=scan_duration,
            flight_time_days=scan_flight_time_bound, 
            dt_step_hours=dt_step_hours,
            origin=self.origin,
            target=self.target
        )
        
        # Store result
        t_iso = opt_launch_dt.isoformat().replace('+00:00', 'Z')
        if not t_iso.endswith('Z'): t_iso += 'Z'
        
        self.t_launch_iso = t_iso
        self.flight_time_days = opt_flight_sec / 86400.0 # Update to optimized time
        self.opt_v_dep = opt_v_dep 
        
        print(f"--> Selected Launch: {self.t_launch_iso} (TOF: {self.flight_time_days:.2f}d)")
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
            
            # Truncate log to stop BEFORE launch time
            # Robust method: Convert to datetime and filter
            truncated_log = []
            
            t_launch_dt = datetime.fromisoformat(best_step['time'].replace('Z', '+00:00'))
            
            for i, step in enumerate(scan_log):
                t_step_dt = datetime.fromisoformat(step['time'].replace('Z', '+00:00'))
                
                # If step is at or after launch, stop (launch state will be start of burn)
                if t_step_dt >= t_launch_dt:
                    print(f"  [Transfer] Truncating Parking Log at index {i} (Time: {step['time']})")
                    break
                if t_step_dt >= t_launch_dt:
                    print(f"  [Transfer] Truncating Parking Log at index {i} (Time: {step['time']})")
                    break
                truncated_log.append(step)
            
            self.add_log(truncated_log)
            
            # Update Launch State
            self.launch_state = best_step
            self.t_launch_iso = best_step['time'] # Updated
            print(f"  Phased Launch Time: {self.t_launch_iso}")
            
        else:
            # Direct Launch state
            pass
            
    def execute_departure(self, thrust: float, isp: float, initial_mass: float, arrival_periapsis_km: float = 0.0):
        """
        Plans and executes the departure burn (Finite Burn).
        arrival_periapsis_km: Altitude of periapsis at arrival (B-Plane targeting).
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
        final_target_pos = np.array(p_target) # Initialize safe default
        
        # Calculate Lambert Arc (Interplanetary/Jovicentric)
        try:
             v_lamb, v_arr_lamb = solve_lambert(r_launch, np.array(p_target), dt_sec, MU_JUP)
             
             # Aimpoint Logic
             final_target_pos = np.array(p_target)
             print(f"  [Debug] arrival_periapsis_km = {arrival_periapsis_km}")
             
             if arrival_periapsis_km > 0:
                 # Calculate V_inf at arrival
                 p_arr, v_arr_body = self.engine.get_body_state(self.target, t_arr_iso)
                 v_inf_vec = v_arr_lamb - np.array(v_arr_body)
                 
                 print(f"  [Debug] V_inf magnitude: {np.linalg.norm(v_inf_vec)}")
                 
                 # Construct B-Plane Vector
                 # Transfer Plane Normal:
                 h_vec = np.cross(r_launch, v_lamb)
                 h_norm = np.linalg.norm(h_vec)
                 h_hat = h_vec / h_norm
                 
                 # B-Vector (in plane) = h_hat cross v_inf_hat
                 v_inf_norm = np.linalg.norm(v_inf_vec)
                 v_inf_hat = v_inf_vec / v_inf_norm
                 
                 b_vec_dir = np.cross(h_hat, v_inf_hat)
                 b_norm = np.linalg.norm(b_vec_dir)
                 print(f"  [Debug] B-Vec Norm: {b_norm}")
                 
                 if b_norm < 1e-9:
                     b_vec_dir = np.array([0,0,1]) # Fallback
                 else:
                     b_vec_dir = b_vec_dir / b_norm
                 
                 # Radius (Periapsis)
                 radius_body = 2410.3 # Callisto default
                 mu_target = self.engine.GM['callisto'] # Default
                 
                 if self.target == 'ganymede': 
                     radius_body = 2634.1
                     mu_target = self.engine.GM['ganymede']
                 
                 r_p = radius_body + arrival_periapsis_km
                 
                 # Impact Parameter (Gravitational Focusing)
                 # b = r_p * sqrt(1 + 2*mu / (r_p * v_inf^2))
                 v_inf_sq = np.dot(v_inf_vec, v_inf_vec)
                 b_mag = r_p * np.sqrt(1 + 2 * mu_target / (r_p * v_inf_sq))
                 
                 print(f"  [Aimpoint] r_p: {r_p:.1f} km, v_inf: {np.sqrt(v_inf_sq):.3f} km/s")
                 print(f"  [Aimpoint] Impact Parameter (b): {b_mag:.1f} km (Focusing Factor: {b_mag/r_p:.2f})")
                 
                 # Offset Target
                 aimpoint = b_vec_dir * b_mag 
                 final_target_pos = np.array(p_target) + aimpoint
                 
                 self.target_pos_vector = final_target_pos # Store for MCC
                 
                 dist_offset = np.linalg.norm(final_target_pos - np.array(p_target))
                 print(f"  [Debug] Target Offset Distance: {dist_offset} km")
                 print(f"  [Transfer] Retargeting to Periapsis: {arrival_periapsis_km:.1f} km")

             
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
             print(f"Lambert Guess Failed: {e}")
             # Fallback
             v_guess = (final_target_pos - r_launch) / dt_sec
             
        # 3. Solve (Impulsive Shooting)
        # Uses JAX Newton-Raphson to hit *exact* target (Center or Offset)
        v_impulse_refined = self.planner.solve_impulsive_shooting(
            r_start=list(r_launch),
            t_start_iso=self.t_launch_iso,
            dt_seconds=dt_sec,
            r_target=list(final_target_pos),
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
        t_burn *= 1.10 # Increased margin for gravity loss
        
        # Heuristic Offset
        t_start_burn_obj = t_launch_obj - timedelta(seconds=t_burn/2.0)
        t_start_burn_iso = t_start_burn_obj.isoformat().replace('+00:00', 'Z')
        
        # Propagate parking to start burn (Back or Fwd)
        p_state_time_obj = datetime.fromisoformat(self.launch_state['time'].replace('Z', '+00:00'))
        
        # Fix: Find exact start state from existing parking log to ensure continuity
        start_state = None
        
        if self.logs and self.logs[-1]:
            parking_log = self.logs[-1]
            # Assumes sorted time
            
            # Find interval t_i <= t_burn_start < t_i+1
            idx_found = -1
            for i in range(len(parking_log)-1):
                t0_str = parking_log[i]['time']
                t1_str = parking_log[i+1]['time']
                t0 = datetime.fromisoformat(t0_str.replace('Z', '+00:00'))
                t1 = datetime.fromisoformat(t1_str.replace('Z', '+00:00'))
                
                if t0 <= t_start_burn_obj <= t1:
                    idx_found = i
                    break
            
            if idx_found != -1:
                state_base = parking_log[idx_found]
                t_base = datetime.fromisoformat(state_base['time'].replace('Z', '+00:00'))
                dt_gap = (t_start_burn_obj - t_base).total_seconds()
                
                if dt_gap > 0.001:
                    # Small propagation to close gap
                    gap_log = self.planner.evaluate_trajectory(
                        r_start=state_base['position'], v_start=state_base['velocity'],
                        t_start_iso=state_base['time'], dt_seconds=dt_gap,
                        mass=initial_mass, n_steps=2
                    )
                    start_state = gap_log[-1]
                else:
                    start_state = state_base
                print(f"  [Departure] Burn Start stitched to Parking Log (Index {idx_found}, Offset {dt_gap:.3f}s)")
        
        if start_state is None:
            # Fallback: Backward propagation if not found (e.g. burn starts before parking log?)
            print("  [Departure] Burn Start outside Parking Log. Propagating backward...")
            dt_prop = (t_start_burn_obj - p_state_time_obj).total_seconds()
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
            target_pos=list(final_target_pos),
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
        self.add_log(burn_log)
        self.burn_end_state = burn_log[-1]
        
        # Log Event
        self.events.append({
            "type": "departure",
            "time": t_start_burn_iso,
            "duration": t_burn,
            "delta_v": dv_mag,
            "delta_v_vec": dv_impulse.tolist()
        })
        
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
        self.add_log(coast_log)
        
        # Plan MCC
        t_arr_obj = t_launch_obj + timedelta(seconds=dt_flight)
        t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
        
        if self.target_pos_vector is not None:
             r_target = self.target_pos_vector
             print("  [MCC] Targeting previously defined Aimpoint (B-Plane).")
        else:
             p_target_body, _ = self.engine.get_body_state(self.target, t_arr_iso)
             r_target = np.array(p_target_body)
        
        result = self.planner.plan_correction_maneuver(
            current_state=coast_log[-1],
            target_pos=list(r_target),
            target_time_iso=t_arr_iso,
            thrust=thrust, isp=isp,
            tolerance_km=10.0,
            previous_state=end_burn
        )
        
        self.add_log(result['maneuver_log'])
        self.last_result = result
        
        # Log Event
        # MCC result usually has dv vector
        dv_vec = result.get('dv_vec', [0,0,0])
        dv_mag = np.linalg.norm(dv_vec)
        
        self.events.append({
            "type": "mcc",
            "time": t_mcc_iso, # Or start of burn?
            "duration": result.get('duration', 0.0), # Assuming planner returns this
            "delta_v": dv_mag,
            "delta_v_vec": list(dv_vec)
        })
        
        return result
        
    def execute_arrival(self, target_ecc: float = 0.0, periapsis_alt_km: float = 500.0):
        """
        Executes capture burn at periapsis.
        """
        print(f"\n[Transfer] executing Arrival/Capture (e={target_ecc}, alt={periapsis_alt_km} km)...")
        
        # 1. Find Periapsis State from last log
        last_log = self.logs[-1]
        
        # Search for min distance
        best_idx = -1
        min_dist = float('inf')
        
        target_key = self.target
        mu_target = self.engine.GM[target_key]
        
        for i, p in enumerate(last_log):
            t_iso = p['time']
            # JAX logs might not have bodies, use engine
            p_body, _ = self.engine.get_body_state(target_key, t_iso)
            
            r_ship = np.array(p['position'])
            r_body = np.array(p_body)
            
            dist = np.linalg.norm(r_ship - r_body)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
                    
        if best_idx == -1:
            print("  ERROR: No approach found in last log.")
            return None
            
        periapsis_state = last_log[best_idx]
        print(f"  Periapsis found at t={periapsis_state['time']}, r={min_dist:.1f} km")
        
        # 2. Calculate Impulsive Burn
        r_ship = np.array(periapsis_state['position'])
        v_ship = np.array(periapsis_state['velocity'])
        
        r_body, v_body = self.engine.get_body_state(target_key, periapsis_state['time'])
        
        dv_vec, dv_mag = self.planner.solve_capture_burn_impulsive(
            r_ship, v_ship, np.array(r_body), np.array(v_body),
            moon_gm=mu_target, target_ecc=target_ecc
        )
        
        print(f"  Required Impulsive DV: {dv_mag*1000:.1f} m/s")
        
        # 3. Finite Burn Parameters
        mass_pre = periapsis_state['mass']
        thrust = 50000.0 # N (High Thrust for instant capture)
        isp = 320.0  # s
        g0 = 9.80665
        ve = isp * g0 / 1000.0 # km/s
        mf = mass_pre / np.exp(abs(dv_mag)/ve)
        m_prop = mass_pre - mf
        t_burn = (m_prop * ve * 1000.0) / thrust 
        
        print(f"  Estimated Burn Duration: {t_burn:.1f} s")
        
        # Use Constant Steering aligned with DV Vector
        u_dir = dv_vec / np.linalg.norm(dv_vec)
        params = list(u_dir)
        
        # FIX: Truncate logs at periapsis to prepare for seamless stitching
        if self.logs:
             # We want to keep everything UP TO best_idx (exclusive, as Capture starts there)
             # But periapsis_state is at best_idx. 
             # So we cut at best_idx.
             print(f"  [Capture] Truncating coast log at index {best_idx} (t={periapsis_state['time']})")
             self.logs[-1] = self.logs[-1][:best_idx]
             if not self.logs[-1]: self.logs.pop()

        # 4. Execute Burn (Constant Steering)
        capture_log = self.planner.evaluate_burn(
             r_start=periapsis_state['position'],
             v_start=periapsis_state['velocity'],
             t_start_iso=periapsis_state['time'],
             dt_seconds=t_burn,
             lts_params=np.array(params), # Constant vector
             thrust=thrust, isp=isp, mass_init=mass_pre,
             steering_mode='constant'
        )
        print(f"  Capture Log: {len(capture_log)} points")
        self.add_log(capture_log)
        
        # Log Event
        self.events.append({
            "type": "capture",
            "time": periapsis_state['time'], # Changed from start_state to periapsis_state
            "duration": t_burn,
            "delta_v": dv_mag,
            "delta_v_vec": dv_vec.tolist(), # Changed from dv_impulse to dv_vec
            "target_ecc": target_ecc
        })
        
        # 6. Parking Orbit
        parking_log = self.planner.evaluate_trajectory(
             r_start=capture_log[-1]['position'],
             v_start=capture_log[-1]['velocity'],
             t_start_iso=capture_log[-1]['time'],
             dt_seconds=86400.0 * 2, # 2 days
             mass=capture_log[-1]['mass'],
             n_steps=500
        )
        print(f"  Parking Log: {len(parking_log)} points. End Time: {parking_log[-1]['time']}")
        self.add_log(parking_log)
        
        return capture_log, parking_log
        
