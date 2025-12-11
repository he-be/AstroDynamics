import numpy as np
from datetime import datetime, timedelta

def stumpff_c(z):
    if z > 0:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z
    elif z < 0:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1) / (-z)
    else:
        return 0.5

def stumpff_s(z):
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / (s**3)
    elif z < 0:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / ((-z)**1.5)
    else:
        return 1.0 / 6.0


def kepler_to_cartesian(a, e, i_rad, Omega_rad, omega_rad, nu_rad, mu):
    """
    Convert Keplerian elements to Cartesian State Vectors.
    
    Args:
        a: Semi-major axis (km)
        e: Eccentricity
        i_rad: Inclination (radians)
        Omega_rad: Longitude of Ascending Node (radians)
        omega_rad: Argument of Periapsis (radians)
        nu_rad: True Anomaly (radians)
        mu: Gravitational parameter
        
    Returns:
        r_vec, v_vec (numpy arrays)
    """
    # 1. Position/Velocity in Perifocal Frame
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu_rad))
    
    # Perifocal coordinates
    # r_orbit = [r cos(nu), r sin(nu), 0]
    # v_orbit = sqrt(mu/p) * [-sin(nu), e + cos(nu), 0]
    
    x_orbit = r * np.cos(nu_rad)
    y_orbit = r * np.sin(nu_rad)
    
    vx_orbit = np.sqrt(mu/p) * (-np.sin(nu_rad))
    vy_orbit = np.sqrt(mu/p) * (e + np.cos(nu_rad))
    
    r_perifocal = np.array([x_orbit, y_orbit, 0.0])
    v_perifocal = np.array([vx_orbit, vy_orbit, 0.0])
    
    # 2. Rotation Matrix to Inertial
    # R3(-Omega) * R1(-i) * R3(-omega)
    # Actually just standard rotation matrix R_perifocal_to_inertial
    
    cO = np.cos(Omega_rad)
    sO = np.sin(Omega_rad)
    cw = np.cos(omega_rad)
    sw = np.sin(omega_rad)
    ci = np.cos(i_rad)
    si = np.sin(i_rad)
    
    # R = [ [cO cw - sO sw ci, -cO sw - sO cw ci,  sO si],
    #       [sO cw + cO sw ci, -sO sw + cO cw ci, -cO si],
    #       [sw si,             cw si,             ci   ] ]
    # Wait, simple matrix multiplication of rotations:
    # R = Rz(Omega) @ Rx(i) @ Rz(omega)
    
    Rz_W = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rx_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    
    R = Rz_W @ Rx_i @ Rz_w
    
    r_vec = R @ r_perifocal
    v_vec = R @ v_perifocal
    
    return r_vec, v_vec

def solve_lambert(r1, r2, dt, mu, cw=False, max_revs=0):
    """
    Solve Lambert's problem using Universal Variables (Bate, Mueller, White).
    Robust implementation using Bisection.
    """
    tm = -1 if cw else 1
    
    r1_vec = np.array(r1)
    r2_vec = np.array(r2)
    r1_mag = np.linalg.norm(r1_vec)
    r2_mag = np.linalg.norm(r2_vec)
    
    # Check for 180 degree transfer
    cross_r1_r2 = np.cross(r1_vec, r2_vec)
    cross_mag = np.linalg.norm(cross_r1_r2)
    
    # ... existing code ...

    dot_prod = np.dot(r1_vec, r2_vec)
    
    if cross_mag < 1e-6 and dot_prod < 0:
        raise ValueError("Lambert Solver: 180 degree transfer singularity.")
        
    # Calculate dnu
    cos_dnu = np.clip(dot_prod / (r1_mag * r2_mag), -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    
    if tm == -1:
        dnu = 2 * np.pi - dnu
        
    # A constant
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(dnu)))
    
    # Time of Flight Function
    def get_tof(z):
        C = stumpff_c(z)
        S = stumpff_s(z)
        
        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
        
        if C <= 0: C = 1e-9 
        
        if y < 1e-9: 
            return 0.0
            
        x = np.sqrt(y / C)
        t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return t

    # Bisection Search
    # Search range.
    low = -100.0
    high = 100.0 # (2pi)^2 ~ 40.
    
    t_low = get_tof(low)
    t_high = get_tof(high)
    
    # Bracket check
    # If dt < t_low (need more hyperbolic), decrease low
    # If dt > t_high (need more elliptic), increase high
    
    for _ in range(20):
        if t_low <= dt <= t_high:
            break
        if dt < t_low:
             low *= 2.0
             # Guard against run away
             if low < -100000: break
             t_low = get_tof(low)
        else:
             high *= 2.0
             if high > 100000: break
             t_high = get_tof(high)
             
    # Bisect
    z = 0.0
    for i in range(100):
        mid = (low + high) / 2.0
        z = mid
        t_mid = get_tof(mid)
        
        if np.isclose(t_mid, dt, rtol=1e-6):
            break
            
        if t_mid < dt: 
            # Need more time -> Increase z (Elliptic)
            # z increases -> C decreases -> y increases?
            # Standard: t(z) is monotonically increasing with z
            low = mid
            t_low = t_mid
        else:
            high = mid
            t_high = t_mid
            
    # Reconstruct
    C = stumpff_c(z)
    S = stumpff_s(z)
    y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
    
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    
    g_dot = 1 - y / r2_mag
    
    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g
    
    return v1, v2

def refine_transfer(engine, r_start, v_guess, t_start, t_end, target_pos_at_end):
    """
    Refines the departure velocity vector using a single-shooting differential correction method
    to hit the target position in the N-body model.
    """
    v_curr = np.array(v_guess, dtype=float)
    dt_obj = datetime.fromisoformat(t_end.replace('Z', '+00:00')) - datetime.fromisoformat(t_start.replace('Z', '+00:00'))
    dt = dt_obj.total_seconds()
    
    print(f"  [Refining Transfer] Targeting {target_pos_at_end}...")
    
    for i in range(25): # Max iterations
        # Propagate nominal
        state0 = np.concatenate([r_start, v_curr]).tolist()
        try:
            # Safer to pass t_eval=[0.0, dt] to ensure we get exactly the end state time
            states = engine.propagate(state0, t_start, dt, t_eval=[0.0, dt])
            if not states:
                print("Refine Transfer: No states returned from propagate.")
                return v_curr
            r_final = np.array(states[-1][:3])
        except Exception as e:
            print(f"Refine Transfer Propagation Error: {e}")
            import traceback
            traceback.print_exc()
            return v_curr # Fail safe
            
        miss_vec = r_final - np.array(target_pos_at_end)
        error = np.linalg.norm(miss_vec)
        
        if error < 500.0: # 500 km tolerance
            return v_curr
            
        # Finite Difference Jacobian
        epsilon = 0.001 # 1 m/s perturbation
        J = np.zeros((3,3))
        
        for j in range(3):
            v_pert = v_curr.copy()
            v_pert[j] += epsilon
            state_pert = np.concatenate([r_start, v_pert]).tolist()
            states_p = engine.propagate(state_pert, t_start, dt, t_eval=[0.0, dt])
            r_final_p = np.array(states_p[-1][:3])
            
            # Column j of J
            col = (r_final_p - r_final) / epsilon
            J[:, j] = col
            
        # Update
        try:
            correction = -np.linalg.solve(J, miss_vec)
            mag = np.linalg.norm(correction)
            if mag > 2.0: 
                correction = correction * (2.0 / mag)
                
            v_curr += correction
        except np.linalg.LinAlgError:
            return v_curr
            
    print(f"    Convergence incomplete (Error {error:.1f} km). Using result.")
    return v_curr

def refine_finite_transfer(engine, r_start, v_start, t_start, t_end, target_pos_at_end, mass, thrust, isp, seed_dv_vec=None, use_variational=True):
    """
    Refines the Delta-V vector for a FINITE burn.
    Iteratively solves for the optimal thrust vector (assumed constant inertial direction)
    that hits the target position after Burn + Coast.
    """
    v_start = np.array(v_start)
    
    # 0. Initial Guess
    dt_obj = datetime.fromisoformat(t_end.replace('Z', '+00:00')) - datetime.fromisoformat(t_start.replace('Z', '+00:00'))
    dt = dt_obj.total_seconds()
    
    if seed_dv_vec is not None:
        dv_curr = np.array(seed_dv_vec, dtype=float)
    else:
        # Simple Lambert for guess
        v1, _ = solve_lambert(np.array(r_start), np.array(target_pos_at_end), dt, engine.GM['jupiter'])
        dv_curr = v1 - v_start
    
    print(f"  [Finite Shooter] Starting. Guess DV: {np.linalg.norm(dv_curr)*1000:.1f} m/s")
    
    params = {'mass': mass, 'thrust': thrust, 'isp': isp, 'g0': 9.80665}
    t_start_dt = datetime.fromisoformat(t_start.replace('Z', '+00:00'))
    
    error = 999999.9
    
    for i in range(25):
        # 1. Calculate Duration from DV Magnitude
        dv_mag_km = np.linalg.norm(dv_curr)
        dv_meters = dv_mag_km * 1000.0
        ve = params['isp'] * params['g0']
        
        # Duration (Tsiolkovsky invert)
        # duration = m0 * ve/F * (1 - exp(-dv/ve))
        duration = (params['mass'] * ve * (1 - np.exp(-dv_meters/ve))) / params['thrust']
        if duration < 0.1: duration = 0.1 
        
        # Thrust Vector
        if dv_mag_km < 1e-6:
             thrust_dir = np.array([1,0,0])
        else:
             thrust_dir = dv_curr / dv_mag_km
             
        thrust_vec = thrust_dir * params['thrust']
        
        # 2. Run Burn & Coast
        r_final = None
        v_final = None
        burn_end_state = None
        burn_end_mass = None
        
        burn_res = None
        coast_res = None
        
        # Branch A: Variational Equations (Analytical)
        if use_variational:
            # Run Burn with STM
            burn_start_state = list(r_start) + list(v_start)
            burn_end_state, burn_end_mass, burn_res  = engine.propagate_controlled(
                 burn_start_state, t_start, duration, thrust_vec.tolist(), params['mass'], params['isp'],
                 with_variational_equations=True
            )
            
            # Coast with STM
            coast_dt = dt - duration
            if coast_dt < 0: coast_dt = 0.0
            t_burn_end = t_start_dt + timedelta(seconds=duration)
            t_burn_end_iso = t_burn_end.isoformat().replace('+00:00', 'Z')
            
            try:
                coast_end_state, _, coast_res = engine.propagate_controlled(
                    burn_end_state, t_burn_end_iso, coast_dt, [0.0, 0.0, 0.0], burn_end_mass, params['isp'],
                    with_variational_equations=True
                )
                r_final = np.array(coast_end_state[:3])
                v_final = np.array(coast_end_state[3:6])
            except Exception as e:
                print(f"Propagation Error (VarEq): {e}")
                return dv_curr

        # Branch B: Standard Propagation (for Finite Diff or just checking error)
        else:
             state0 = list(r_start) + list(v_start)
             burn_end_state, burn_end_mass = engine.propagate_controlled(
                 state0, t_start, duration, thrust_vec.tolist(), params['mass'], params['isp'],
                 with_variational_equations=False
             )
             coast_dt = dt - duration
             if coast_dt < 0: coast_dt = 0.0
             t_burn_end = t_start_dt + timedelta(seconds=duration)
             t_burn_end_iso = t_burn_end.isoformat().replace('+00:00', 'Z')
             
             try:
                 coast_states = engine.propagate(burn_end_state, t_burn_end_iso, coast_dt, t_eval=[0.0, coast_dt])
                 r_final = np.array(coast_states[-1][:3])
             except Exception:
                 return dv_curr
        
        # 4. Error
        miss_vec = r_final - np.array(target_pos_at_end)
        error = np.linalg.norm(miss_vec)
        
        if error < 200.0:
             print(f"  [Finite Shooter] Converged! Iter {i}, Error {error:.1f} km")
             return dv_curr
             
        # 5. Jacobian Calculation
        J = None
        
        if use_variational:
            # Analytical Jacobian
            # Determine N from length L: L = N + N^2 + 3N = N^2 + 4N
            L = len(burn_res)
            import math
            N = int(-2 + math.sqrt(4 + L))
            
            # Helper to extract Matrix Blocks
            def extract_matrices(flat_res):
                stm_flat = flat_res[N : N + N*N]
                stm = stm_flat.reshape((N, N))
                s_param_flat = flat_res[N + N*N :]
                s_param = s_param_flat.reshape((3, N)).T
                return stm, s_param

            # A. Get d_rf/d_T
            _, S_burn = extract_matrices(burn_res)
            Phi_coast, _ = extract_matrices(coast_res)
            
            # Ship Position indices
            # State vector ends with [..., px, py, pz, vx, vy, vz, mass]
            # px is at N-7, py at N-6, pz at N-5
            ship_pos_idx = [N - 7, N - 6, N - 5]
            
            Phi_coast_r = Phi_coast[ship_pos_idx, :] # 3 x N
            dr_dT = Phi_coast_r @ S_burn # 3 x 3
            
            # B. d_T / d_dv
            dv_mag = np.linalg.norm(dv_curr)
            u = dv_curr / dv_mag
            F = params['thrust']
            I = np.eye(3)
            dT_ddv = (F / dv_mag) * (I - np.outer(u, u))
            
            term1 = dr_dT @ dT_ddv
            
            # C. Term 2: d_rf/d_dur * d_dur/d_dv
            m0 = params['mass']
            dv_mag_m = dv_mag * 1000.0
            exp_factor = np.exp(-dv_mag_m / ve)
            d_dur_d_mag_m = (m0 / F) * exp_factor
            
            d_dur_d_dv = (d_dur_d_mag_m * 1000.0) * u
            
            # d_rf/d_dur Calculation
            # Phi_rv: Rows Pos, Cols Vel of Ship
            # Ship Pos: N-7 to N-4
            # Ship Vel: N-4 to N-1
            Phi_rv = Phi_coast[N-7:N-4, N-4:N-1]
            a_thrust_end = thrust_vec / burn_end_mass
            d_rf_d_dur = Phi_rv @ a_thrust_end
            
            term2 = np.outer(d_rf_d_dur, d_dur_d_dv)
            
            J = term1 + term2
            
        else:
            # Finite Difference Jacobian
            J = np.zeros((3,3))
            eps = 0.001 # 1 m/s perturbation
            state0 = list(r_start) + list(v_start)
            
            for k in range(3):
                dv_pert = dv_curr.copy()
                dv_pert[k] += eps
                
                # Recalculate duration/thrust
                dv_p_mag = np.linalg.norm(dv_pert)
                dur_p = (params['mass'] * ve * (1 - np.exp(-dv_p_mag*1000.0/ve))) / params['thrust']
                if dur_p < 0.1: dur_p = 0.1
                
                t_dir_p = dv_pert / dv_p_mag if dv_p_mag > 1e-6 else np.array([1,0,0])
                t_vec_p = t_dir_p * params['thrust']
                
                # Propagate Burn
                bs_p, bm_p = engine.propagate_controlled(
                    state0, t_start, dur_p, t_vec_p.tolist(), params['mass'], params['isp'],
                    with_variational_equations=False
                )
                
                # Propagate Coast
                c_dt_p = dt - dur_p
                if c_dt_p < 0: c_dt_p = 0.0
                t_be_p = t_start_dt + timedelta(seconds=dur_p)
                t_be_iso_p = t_be_p.isoformat().replace('+00:00', 'Z')
                
                try:
                    cs_p = engine.propagate(bs_p, t_be_iso_p, c_dt_p, t_eval=[0.0, c_dt_p])
                    rp_final = np.array(cs_p[-1][:3])
                except:
                    rp_final = r_final # Fallback
                
                col = (rp_final - r_final) / eps
                J[:, k] = col
            
        # 6. Update
        try:
             # Regularize? damped least squares?
             correction = -np.linalg.solve(J, miss_vec)
             # Clamp correction
             mag = np.linalg.norm(correction)
             if mag > 0.1: correction *= (0.1/mag) # Max 100 m/s step
             
             print(f"    Step: {mag*1000:.1f} m/s -> Limit {np.linalg.norm(correction)*1000:.1f} m/s")
             dv_curr += correction
        except np.linalg.LinAlgError:
             return dv_curr
             
    print(f"  [Finite Shooter] Max iters. Error {error:.1f} km")
    return dv_curr

def refine_lts_transfer(engine, r_start, v_start, t_start, t_end, target_pos_at_end, mass, thrust, isp, seed_dv_vec=None):
    """
    Refines transfer using Linear Tangent Steering (LTS).
    Optimizes bilinear tangent parameters [ax, ay, az, bx, by, bz].
    u(t) = unit(a + b*t).
    """
    control_params = np.zeros(6)
    # Init from seed (constant direction assumption)
    if seed_dv_vec is not None:
         dv_norm = np.linalg.norm(seed_dv_vec)
         if dv_norm > 1e-6:
             control_params[0:3] = np.array(seed_dv_vec) / dv_norm
         else:
             control_params[0] = 1.0
    else:
         control_params[0] = 1.0 # Default X direction
         
    dt_obj = datetime.fromisoformat(t_end.replace('Z', '+00:00')) - datetime.fromisoformat(t_start.replace('Z', '+00:00'))
    dt = dt_obj.total_seconds()
    
    print(f"  [LTS Refinement] Targeting {target_pos_at_end} with LTS...")
    
    for i in range(20):
        # 1. Propagate with Variational Equations (State + Jacobian)
        try:
            res_list, final_mass, J = engine.propagate_controlled(
                list(r_start) + list(v_start), t_start, dt, list(control_params), mass, isp,
                steering_mode='linear_tangent', thrust_magnitude=thrust,
                with_variational_equations=True
            )
        except Exception as e:
            print(f"Propagate failed or VarEq error: {e}")
            break
            
        r_p = np.array(res_list[0:3])
        miss = r_p - np.array(target_pos_at_end)
        err_norm = np.linalg.norm(miss)
        
        print(f"    Iter {i}: Error = {err_norm:.1f} km")
        
        if err_norm < 10.0: # Convergence check
            return control_params, r_p, final_mass
            
        # 2. Update (Min Norm)
        try:
            dx = -np.linalg.pinv(J) @ miss
            
            # Limit step
            mag = np.linalg.norm(dx)
            if mag > 0.5: dx = dx * (0.5 / mag)
            
            control_params += dx
            
        except np.linalg.LinAlgError:
            break
            
    return control_params

def predict_hohmann_window_circular(r1, r2, current_phase_rad, mu):
    """
    Calculates time to next Hohmann opportunity for circular coplanar orbits.
    
    Args:
        r1: Origin orbital radius (km)
        r2: Target orbital radius (km)
        current_phase_rad: theta_2 - theta_1 (radians, range -pi to pi)
                           Angle of Target relative to Origin.
        mu: Gravitational parameter of central body
        
    Returns:
        dt_wait: Seconds to wait for launch
        dt_flight: Seconds of flight time
        phi_req: Required phase angle (Target - Origin) at launch (rad)
    """
    import math
    
    # Mean Motions (rad/s)
    n1 = math.sqrt(mu / r1**3)
    n2 = math.sqrt(mu / r2**3)
    
    # Relative Rate (Target relative to Origin)
    n_rel = n2 - n1 
    
    # Hohmann Transfer Time (Semi-ellipse)
    a_trans = (r1 + r2) / 2.0
    t_flight = math.pi * math.sqrt(a_trans**3 / mu)
    
    # Required Phase Alignment at Launch
    # We want to arrive when Target is at Apoapsis/Periapsis (180 deg from Launch).
    # Spacecraft travels 180 deg (pi).
    # Target travels n2 * t_flight.
    # At arrival: theta_2 = theta_1 (mod 2pi) ? No, they meet.
    # theta_SC_final = theta_1_initial + pi
    # theta_2_final = theta_2_initial + n2 * t_flight
    # We want theta_SC_final == theta_2_final
    # theta_1_initial + pi = theta_2_initial + n2 * t_flight
    # theta_2_initial - theta_1_initial = pi - n2 * t_flight
    
    phi_req = math.pi - n2 * t_flight
    
    # Normalize phases to (-pi, pi)
    def normalize(angle):
        while angle > math.pi: angle -= 2*math.pi
        while angle <= -math.pi: angle += 2*math.pi
        return angle
        
    phi_req = normalize(phi_req)
    phi_curr = normalize(current_phase_rad)
    
    # We need to span the difference d_phi at rate n_rel.
    # phi_final = phi_start + rate * t
    # phi_req = phi_curr + n_rel * t + 2pi*k
    # n_rel * t = phi_req - phi_curr + 2pi*k
    # t = (phi_req - phi_curr + 2pi*k) / n_rel
    
    d_phi = phi_req - phi_curr
    
    # Find smallest positive t
    # Since n_rel is negative (inner->outer), dividing by negative flips inequality
    
    candidates = []
    # Test k range [-3 to 3]
    for k in range(-5, 6):
        numerator = d_phi + 2*math.pi*k
        t = numerator / n_rel
        if t >= 0:
            candidates.append(t)
            
    if not candidates:
        dt_wait = 0.0
    else:
        dt_wait = min(candidates)
        
    return dt_wait, t_flight, phi_req
