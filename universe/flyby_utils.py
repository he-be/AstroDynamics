import numpy as np
from datetime import datetime, timedelta
from planning import solve_lambert

class FlybyCandidateFinder:
    def __init__(self, engine):
        self.engine = engine
        self.mu_primary = engine.GM['jupiter']
        
    def find_candidates(self, body_start, body_end, intermediates, 
                       t_start_iso, search_days=30.0, 
                       flight_time_1_range=(1.0, 5.0),
                       flight_time_2_range=(1.0, 5.0),
                       max_v_inf=8.0):
        """
        Search for Flyby candidates by matching V_inf magnitude at the intermediate body.
        
        Args:
            max_v_inf: Max allowed V_inf (km/s) to filter high-energy solutions.
        """
        candidates = []
        
        start_dt = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
        
        # Grid parameters
        step_days = 2.0 # Coarse
        
        print(f"Searching candidates from {body_start} to {body_end} via {intermediates}...")
        
        for mid in intermediates:
            print(f"  Checking {mid} (Max V_inf: {max_v_inf} km/s)...")
            # Grid Search
            # We iterate T_dep
            for i in range(int(search_days / step_days)):
                t_dep = start_dt + timedelta(days=i*step_days)
                t_dep_iso = t_dep.isoformat().replace('+00:00', 'Z')
                
                p_start, v_start_body = self.engine.get_body_state(body_start, t_dep_iso)
                
                # Iterate T_flyby (Leg 1 Duration)
                ft1_min, ft1_max = flight_time_1_range
                for ft1 in np.linspace(ft1_min, ft1_max, 5): # 5 steps
                    t_flyby = t_dep + timedelta(days=ft1)
                    t_flyby_iso = t_flyby.isoformat().replace('+00:00', 'Z')
                    
                    p_mid, v_mid_body = self.engine.get_body_state(mid, t_flyby_iso)
                    
                    # Solve Leg 1 Lambert
                    try:
                        v1_dep, v1_arr = solve_lambert(np.array(p_start), np.array(p_mid), ft1*86400, self.mu_primary)
                    except:
                        continue
                        
                    # Calculate V_inf_in
                    v_inf_in = v1_arr - np.array(v_mid_body)
                    v_inf_in_mag = np.linalg.norm(v_inf_in)
                    
                    if v_inf_in_mag > max_v_inf:
                        # print(f"  Reject: Inbound V_inf {v_inf_in_mag:.2f} > {max_v_inf}")
                        continue
                    
                    # Iterate T_arr (Leg 2 Duration)
                    ft2_min, ft2_max = flight_time_2_range
                    for ft2 in np.linspace(ft2_min, ft2_max, 5):
                        t_arr = t_flyby + timedelta(days=ft2)
                        t_arr_iso = t_arr.isoformat().replace('+00:00', 'Z')
                        
                        p_end, v_end_body = self.engine.get_body_state(body_end, t_arr_iso)
                        
                        # Solve Leg 2 Lambert
                        try:
                            v2_dep, v2_arr = solve_lambert(np.array(p_mid), np.array(p_end), ft2*86400, self.mu_primary)
                        except:
                            continue
                            
                        # Calculate V_inf_out
                        v_inf_out = v2_dep - np.array(v_mid_body)
                        v_inf_out_mag = np.linalg.norm(v_inf_out)
                        
                        if v_inf_out_mag > max_v_inf:
                            continue
                        
                        # Calculate Max Bending Angle
                        # delta_max = 2 * asin(1 / (1 + (rp * v_inf^2 / mu_moon)))
                        # Use average V_inf for turn capacity estimation
                        v_inf_avg = 0.5 * (v_inf_in_mag + v_inf_out_mag)
                        
                        mu_body = self.engine.GM[mid]
                        # Get physical radius
                        # Radius dict or ask engine? Engine has GM but maybe not radius in simple dict?
                        # Using近似 constraint: Europa=1560km, Gan=2634, Io=1821, Cal=2410
                        radii = {'io': 1821.0, 'europa': 1560.8, 'ganymede': 2634.1, 'callisto': 2410.3}
                        r_body = radii.get(mid, 2000.0) 
                        h_safe = 100.0 # Safe flyby altitude km
                        rp = r_body + h_safe
                        
                        e_hyp = 1 + (rp * v_inf_avg**2 / mu_body)
                        delta_max = 2 * np.arcsin(1 / e_hyp)
                        
                        # Required Turn
                        dot = np.dot(v_inf_in, v_inf_out)
                        cos_theta = dot / (v_inf_in_mag * v_inf_out_mag)
                        theta_req = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                        
                        mismatch = abs(v_inf_in_mag - v_inf_out_mag)
                        
                        # Powered Flyby Cost Estimation
                        # Vector difference (Impulsive change at infinity) - Naive
                        # dv_naive = np.linalg.norm(v_inf_out - v_inf_in)
                        
                        # Better estimation: Maneuver at Periapsis
                        # We need to change V_inf vector from In to Out.
                        # Gravity gives us 'delta_max' (or less) for free.
                        # If theta_req > delta_max, we pay for the difference? 
                        # Or, we just calculate the vector difference between V_p_in and V_p_out?
                        # 
                        # Simplified Powered Flyby Model:
                        # V_p_in_mag = sqrt(v_inf_in^2 + 2*mu/rp)
                        # V_p_out_mag = sqrt(v_inf_out^2 + 2*mu/rp)
                        # The turn 'theta_req' must be achieved during the passage.
                        # This is a complex optimization. 
                        # Simple Metric:
                        # If theta_req <= delta_max: Cost is roughly abs(V_p_out - V_p_in) (Energy change only)
                        # If theta_req > delta_max: We need extra deflection.
                        #
                        # Let's use the Vector Difference of V_inf as a "Upper Bound / Honest Cost" for now,
                        # but subtracted by the "Max Gravity Turn" capability?
                        #
                        # Actually, `europa_slingshot.py` executes `dv_flyby = v_dep_2 - v_arr_1`.
                        # This is the vector difference of velocities *at the body's position in the frame*.
                        # Which IS `v_inf_out - v_inf_in`.
                        # This confirms the current script is doing a "Deep Space Maneuver at the location of Europa"
                        # rather than a true hyperbolic flyby.
                        # A true flyby gains dV = 2*V_inf*sin(delta/2).
                        # Our `dv_flyby` matches the vectors.
                        # If we execute this at Europa, we ARE paying the full vector difference.
                        # The "Gravity Assist" only works if we model the hyperbola.
                        #
                        # BUT, for the Planner, we want to find candidates where this Vector Difference is MINIMIZED 
                        # if we assume we ARE doing a powered flyby.
                        # Wait, if we effectively just do a Deep Space Maneuver (DSM) at Europa's position,
                        # Europa's gravity doesn't help unless we integrate the flyby.
                        # The current `europa_slingshot.py` controller COASTS. It does not integrate the flyby.
                        # It executes `execute_burn` instantaneously.
                        # So the physics engine *sees* a change in velocity.
                        # It does NOT see a gravity turn unless the controller simulates the hyperbolic passage.
                        #
                        # The USER COMPLAINT is that the result (8km/s) is awful.
                        # To fix this, we must select a candidate where `v_inf_out - v_inf_in` is SMALL.
                        # The current Finder sorts by `mismatch` (magnitude diff).
                        # But gives huge `turn_angle`.
                        #
                        # We should sort by `dv_vector_diff = norm(v_inf_out - v_inf_in)`.
                        # This minimizes the burn required *in the current simplified execution model*.
                        
                        dv_total_flyby = np.linalg.norm(v_inf_out - v_inf_in)
                        
                        # Filter massive burns
                        if dv_total_flyby < 10.0: # Allow high burns for comparison
                            candidates.append({
                                'intermediate': mid,
                                't_dep': t_dep_iso,
                                't_flyby': t_flyby_iso,
                                't_arr': t_arr_iso,
                                'v_inf_in': v_inf_in_mag,
                                'v_inf_out': v_inf_out_mag,
                                'mismatch': mismatch,
                                'turn_angle_deg': np.degrees(theta_req),
                                'total_flyby_dv': dv_total_flyby
                            })
                            
        # Sort by TOTAL vector difference (the actual burn cost in current model)
        candidates.sort(key=lambda x: x['total_flyby_dv'])
        return candidates
