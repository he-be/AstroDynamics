
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import json

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def run_jax_lts_scenario():
    print("=== US-10: Ganymede Launch Window Search & LTS ===")
    
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    
    # 1. Mission Context
    mission_start_iso = "2025-07-29T12:00:00Z"
    flight_time_days = 4.0
    scan_window_days = 20.0
    
    print(f"Mission Start: {mission_start_iso}")
    print(f"Scanning {scan_window_days} days for optimal window...")
    
    # 2. Find Optimal Window
    opt_launch_dt, opt_v_dep = jax_planner.find_optimal_launch_window(
        t_window_start_iso=mission_start_iso,
        window_duration_days=scan_window_days,
        flight_time_days=flight_time_days,
        dt_step_hours=6.0
    )
    
    t_launch_iso = opt_launch_dt.isoformat()
    # Ensure Z format
    if '+' in t_launch_iso:
        t_launch_iso = t_launch_iso.split('+')[0] + 'Z'
    elif not t_launch_iso.endswith('Z'):
        t_launch_iso += 'Z'
    
    print(f"--> Selected Launch: {t_launch_iso}")
    
    # Calculate Wait Time
    start_dt_obj = datetime.fromisoformat(mission_start_iso.replace('Z', '+00:00'))
    launch_dt_obj = datetime.fromisoformat(t_launch_iso.replace('Z', '+00:00'))
    
    wait_sec = (launch_dt_obj - start_dt_obj).total_seconds()
    print(f"Parking Duration: {wait_sec/86400:.2f} days")
    
    # 3. Simulate Parking Orbit (Low Ganymede Orbit)
    print("Simulating Parking Orbit...")
    p_gan, v_gan = engine.get_body_state('ganymede', mission_start_iso)
    
    # 500km altitude circular orbit
    R_gan = 2634.1
    alt = 500.0
    r_mag = R_gan + alt
    mu_gan = engine.GM['ganymede']
    v_circ_mag = np.sqrt(mu_gan / r_mag)
    
    # Simple planar orbit (Ganymede-centered)
    # We need to map this to JAX Engine frame (Jupiter-centered)
    # r_ship = r_Gan + r_park
    # v_ship = v_Gan + v_park
    r_park_local = np.array([r_mag, 0.0, 0.0])
    v_park_local = np.array([0.0, v_circ_mag, 0.0])
    
    r_park_start = np.array(p_gan) + r_park_local
    v_park_start = np.array(v_gan) + v_park_local
    
    mass = 1000.0
    
    parking_log = []
    if wait_sec > 60.0:
        # We assume evaluate_trajectory handles JAX propagation
        # Ephemeris resolution: using coarser step for parking is fine? 
        # Ganymede orbit period ~high res needed for stability?
        # Ganymede period: 2*pi*sqrt(r^3/mu) ~ 2*pi*sqrt(3134^3 / 9887) ~ 5500s (~1.5h).
        # We need step size << 1.5h. 300s (5 min) is ok.
        parking_log = jax_planner.evaluate_trajectory(
            r_start=list(r_park_start),
            v_start=list(v_park_start),
            t_start_iso=mission_start_iso,
            dt_seconds=wait_sec,
            mass=mass,
            n_steps=max(200, int(wait_sec/300.0))
        )
        last_park = parking_log[-1]
        r_launch = np.array(last_park['position'])
        v_launch = np.array(last_park['velocity'])
        m_launch = last_park['mass']
        
        # Plot checks: Is parking stable?
        # JAX Engine includes Jupiter gravity (tidal forces). LGO is stable-ish for days.
    else:
        print("Immediate Launch.")
        r_launch = r_park_start
        v_launch = v_park_start
        m_launch = mass

    # 3. Simulate Parking Orbit
    # ...
    
    # Phase Matching Logic
    print("Optimizing Parking Phase...")
    state_curr = parking_log[-1]
    r_curr = np.array(state_curr['position'])
    v_curr = np.array(state_curr['velocity'])
    
    # Calculate Period
    p_gan_curr, _ = jax_planner.engine.get_body_state('ganymede', state_curr['time'])
    r_rel_mag = np.linalg.norm(r_curr - np.array(p_gan_curr))
    mu_gan = jax_planner.engine.GM['ganymede']
    period = 2 * np.pi * np.sqrt(r_rel_mag**3 / mu_gan)
    print(f"  Orbit Period: {period/60.0:.1f} mins")
    
    # Propagate for 1.2 periods to find best release point
    scan_log = jax_planner.evaluate_trajectory(
        r_start=list(r_curr), v_start=list(v_curr),
        t_start_iso=state_curr['time'],
        dt_seconds=period * 1.2,
        mass=m_launch,
        n_steps=60 # Roughly 1-2 min steps
    )
    
    from universe.planning import solve_lambert
    mu_jup = jax_planner.engine.GM['jupiter']
    dt_flight_sec = flight_time_days * 86400.0
    
    best_dv = float('inf')
    best_step = state_curr
    
    for step in scan_log:
        r_try = np.array(step['position'])
        v_try = np.array(step['velocity'])
        t_try_obj = datetime.fromisoformat(step['time'].replace('Z', '+00:00'))
        
        t_arr_try = t_try_obj + timedelta(days=flight_time_days)
        t_arr_try_iso = t_arr_try.isoformat().replace('+00:00', 'Z')
        p_cal_try, _ = jax_planner.engine.get_body_state('callisto', t_arr_try_iso)
        
        try:
             v_lamb, _ = solve_lambert(r_try, np.array(p_cal_try), dt_flight_sec, mu_jup)
             dv = np.linalg.norm(v_lamb - v_try)
             if dv < best_dv:
                 best_dv = dv
                 best_step = step
        except:
             pass

    print(f"  Phase Match found! DV: {best_dv*1000:.1f} m/s")
    
    # Use Best Phase State as Launch
    r_launch = np.array(best_step['position'])
    v_launch = np.array(best_step['velocity'])
    t_launch_iso = best_step['time']
    launch_dt_obj = datetime.fromisoformat(t_launch_iso.replace('Z', '+00:00'))
    
    # Append scan portion to parking log for visualization?
    # Simple concatenation (might be slightly redundant pts but ok)
    parking_log.extend(scan_log[:scan_log.index(best_step)+1])
    
    print(f"Parking End State (Phased): r={r_launch}, v={v_launch}")
    
    t_arrive_iso = (launch_dt_obj + timedelta(days=flight_time_days)).isoformat().replace('+00:00', 'Z')
    p_cal, _ = jax_planner.engine.get_body_state('callisto', t_arrive_iso)
    
    # NEW: Re-calculate Lambert guess
    guess_v = list(opt_v_dep) # Default from search
    try:
        v_lambert_fresh, _ = solve_lambert(r_launch, np.array(p_cal), dt_flight_sec, mu_jup)
        
        # Oberth Correction!
        # Lambert gives V_inf (relative to Ganymede, effectively).
        # We need to add escape energy.
        # v_gan_launch = v_launch # Already have this (Gan velocity at launch time... NO! v_launch is PARKING velocity).
        # v_launch is total velocity. 
        # v_gan_body is Ganymede velocity.
        p_gan_launch, v_gan_body = jax_planner.engine.get_body_state('ganymede', t_launch_iso)
        
        v_inf_vec = v_lambert_fresh - np.array(v_gan_body)
        v_inf_sq = np.linalg.norm(v_inf_vec)**2
        
        r_rel = np.linalg.norm(r_launch - np.array(p_gan_launch))
        mu_gan = jax_planner.engine.GM['ganymede']
        
        v_esc_local_sq = 2 * mu_gan / r_rel
        v_needed_mag = np.sqrt(v_inf_sq + v_esc_local_sq)
        
        # Direction: Ideally aligned with V_inf? 
        # Or aligned with V_park (tangential)?
        # If Phase Matched, V_park is aligned with V_inf roughly.
        # But V_inf direction is the asymptote.
        # Hyperbola vertex velocity direction is same as V_inf? NO.
        # At periapsis, velocity is horizontal. Asymptote is angled.
        # But for high energy, they are close.
        # Let's use V_inf direction as a robust "Outward" guess.
        
        u_inf = v_inf_vec / np.linalg.norm(v_inf_vec)
        v_guess_oberth = np.array(v_gan_body) + v_needed_mag * u_inf
        
        print(f"  V_inf: {np.sqrt(v_inf_sq)*1000:.1f} m/s")
        print(f"  V_esc_local: {np.sqrt(v_esc_local_sq)*1000:.1f} m/s")
        print(f"  V_needed: {v_needed_mag*1000:.1f} m/s")
        
        guess_v = list(v_guess_oberth)
        print("  Using Oberth-Corrected Guess.")

    except Exception as e:
        print(f"Lambert/Oberth failed {e}, using fallback.")
        # If Lambert fails, fall back to a linear guess
        v_guess_lin = (np.array(p_cal) - r_launch) / dt_flight_sec
        guess_v = list(v_guess_lin)
        pass
        
    print("Refining Impulse Target from Parking Orbit...")
    v_impulse_refined = jax_planner.solve_impulsive_shooting(
        r_start=list(r_launch),
        t_start_iso=t_launch_iso,
        dt_seconds=dt_flight_sec,
        r_target=list(p_cal),
        initial_v_guess=guess_v # Use Fresh Guess
    )
    
    dv_impulse = np.array(v_impulse_refined) - v_launch
    dv_mag = np.linalg.norm(dv_impulse)
    print(f"Refined Impulse DV: {dv_mag*1000:.1f} m/s")

    # Burn Sizing
    thrust = 2000.0
    isp = 3000.0
    g0 = 9.80665
    ve = isp * g0 / 1000.0
    m_dot = thrust / (ve * 1000.0)
    
    t_burn_ideal = (m_launch * (1.0 - np.exp(-dv_mag/ve))) / m_dot
    t_burn_ideal = (m_launch * (1.0 - np.exp(-dv_mag/ve))) / m_dot
    t_burn = t_burn_ideal * 1.01 
    
    # 5. Half-Burn Offset Heuristic
    print(f"Applying Half-Burn Offset: shift start by -{t_burn/2:.1f} s")
    start_burn_dt = launch_dt_obj - timedelta(seconds=t_burn/2.0)
    t_start_burn_iso = start_burn_dt.isoformat().replace('+00:00', 'Z')
    
    # Propagate from Mission Start to t_start_burn to get precise bounds
    p0 = parking_log[0]
    t0_obj = datetime.fromisoformat(p0['time'].replace('Z', '+00:00'))
    dt_to_burn_start = (start_burn_dt - t0_obj).total_seconds()
    
    # Check if dt_to_burn_start is positive
    if dt_to_burn_start < 0:
        # Edge case: burn starts before mission start? Unlikely with 10 day wait.
        dt_to_burn_start = 0.0 # Force start
    
    pre_burn_log = jax_planner.evaluate_trajectory(
        r_start=p0['position'], v_start=p0['velocity'],
        t_start_iso=p0['time'],
        dt_seconds=dt_to_burn_start,
        mass=m_launch,
        n_steps=100
    )
    
    r_burn_start = pre_burn_log[-1]['position']
    v_burn_start = pre_burn_log[-1]['velocity']
    
    # Coast Duration: dt_flight - t_burn/2
    dt_coast = dt_flight_sec - t_burn / 2.0
    
    # DEBUG LOGGING
    with open("debug_lts.log", "w") as f:
        f.write(f"Refined Impulse DV: {dv_mag*1000:.1f} m/s\n")
        f.write(f"Mass: {m_launch:.1f} kg\n")
        f.write(f"Thrust: {thrust} N, Isp: {isp} s\n")
        f.write(f"Ve: {ve:.2f} km/s, Flow Rate: {m_dot:.4f} kg/s\n")
        f.write(f"Burn Time: {t_burn:.2f} s\n")
        f.write(f"Half-Burn Offset Applied.\n")
        f.write(f"Launch Time (Impulse): {t_launch_iso}\n")
        f.write(f"Burn Start Time: {t_start_burn_iso}\n")
        f.write(f"Launch Pos: {r_launch}\n")
        f.write(f"Burn Start Pos: {r_burn_start}\n")
        f.write(f"Target Pos: {p_cal}\n")
        f.write(f"Impulse Vector: {dv_impulse}\n")
    
    # 5. Solve LTS
    print("Solving Finite Burn...")
    params_lts = jax_planner.solve_finite_burn_coast(
        r_start=list(r_burn_start),
        v_start=list(v_burn_start),
        t_start_iso=t_start_burn_iso,
        t_burn_seconds=t_burn,
        t_coast_seconds=dt_coast,
        target_pos=list(p_cal),
        mass_init=m_launch,
        thrust=thrust,
        isp=isp,
        impulse_vector=dv_impulse,
        tol_km=100.0
    )

    # 6. Evaluate Burn
    burn_log = jax_planner.evaluate_burn( 
         r_start=list(r_burn_start),
         v_start=list(v_burn_start),
         t_start_iso=t_start_burn_iso,
         dt_seconds=t_burn,
         lts_params=params_lts,
         thrust=thrust,
         isp=isp,
         mass_init=m_launch
    )
    
    # 7. Mid-Course Correction (MCC)
    # Define MCC time relative to Launch
    t_mcc_obj = launch_dt_obj + timedelta(seconds=dt_flight_sec * 0.5)
    t_mcc_iso = t_mcc_obj.isoformat().replace('+00:00', 'Z')
    
    end_burn = burn_log[-1]
    t_burn_end_obj = datetime.fromisoformat(end_burn['time'].replace('Z', '+00:00'))
    dt_coast1 = (t_mcc_obj - t_burn_end_obj).total_seconds()
    
    print(f"Propagating Coast 1 (to MCC): {dt_coast1/3600:.2f} hours")
    
    coast1_log = jax_planner.evaluate_trajectory(
        r_start=end_burn['position'],
        v_start=end_burn['velocity'],
        t_start_iso=end_burn['time'],
        dt_seconds=dt_coast1,
        mass=end_burn['mass'],
        n_steps=100
    )
    
    state_mcc_start = coast1_log[-1]
    r_mcc = np.array(state_mcc_start['position'])
    v_mcc = np.array(state_mcc_start['velocity'])
    m_mcc = state_mcc_start['mass']
    
    print(f"Designing MCC at {t_mcc_iso}...")
    
    # Target (Callisto Arrival)
    t_arr_obj = launch_dt_obj + timedelta(seconds=dt_flight_sec)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    p_cal_arr, _ = jax_planner.engine.get_body_state('callisto', t_arr_iso)
    dt_remaining = (t_arr_obj - t_mcc_obj).total_seconds()
    
    # Solve Lambert for MCC
    skip_tcm2 = False
    try:
        # --- EVALUATION 1: Initial Burn Accuracy ---
        print("  [Evaluation] Checking Initial Burn Accuracy...")
        t_end_burn_obj = datetime.fromisoformat(end_burn['time'].replace('Z', '+00:00'))
        dt_check_1 = (t_arr_obj - t_end_burn_obj).total_seconds()
        check_1_log = jax_planner.evaluate_trajectory(
            r_start=end_burn['position'], v_start=end_burn['velocity'],
            t_start_iso=end_burn['time'], dt_seconds=dt_check_1,
            mass=end_burn['mass'], n_steps=100
        )
        r_check_1 = np.array(check_1_log[-1]['position'])
        err_1 = np.linalg.norm(r_check_1 - np.array(p_cal_arr))
        print(f"  Initial Burn Residual Error: {err_1:.1f} km")

        # Solve Lambert for MCC (Estimate)
        v_mcc_dep, _ = solve_lambert(r_mcc, np.array(p_cal_arr), dt_remaining, mu_jup)
        dv_est = v_mcc_dep - v_mcc
        dv_mag_est = np.linalg.norm(dv_est)
        print(f"  MCC Est Delta-V: {dv_mag_est*1000:.2f} m/s")
        
        # Estimate Burn Time
        m_dot = thrust / (ve * 1000.0)
        t_burn_mcc = (m_mcc * (1.0 - np.exp(-dv_mag_est/ve))) / m_dot
        if t_burn_mcc < 1.0: t_burn_mcc = 1.0
        
        # --- HEURISTIC: Half-Burn Offset & Re-Targeting (N-Body) ---
        print("  [Heuristic] Applying Half-Burn Offset & N-Body Optimization for TCM-1...")
        t_mcc_start_obj = t_mcc_obj - timedelta(seconds=t_burn_mcc/2.0)
        t_mcc_start_iso = t_mcc_start_obj.isoformat().replace('+00:00', 'Z')
        
        # Re-Propagate Coast 1 to start time
        dt_coast1_rev = (t_mcc_start_obj - t_burn_end_obj).total_seconds()
        
        coast1_log = jax_planner.evaluate_trajectory(
            r_start=end_burn['position'], v_start=end_burn['velocity'],
            t_start_iso=end_burn['time'], dt_seconds=dt_coast1_rev,
            mass=end_burn['mass'], n_steps=100
        )
        state_burn_start = coast1_log[-1]
        r_start = list(state_burn_start['position'])
        v_start = list(state_burn_start['velocity'])
        m_start = state_burn_start['mass']
        
        # Solve Finite Burn (Optimizer handles N-Body)
        dt_coast2_est = (t_arr_obj - (t_mcc_start_obj + timedelta(seconds=t_burn_mcc))).total_seconds()
        
        params_mcc = jax_planner.solve_finite_burn_coast(
            r_start=r_start, v_start=v_start,
            t_start_iso=t_mcc_start_iso,
            t_burn_seconds=t_burn_mcc,
            t_coast_seconds=dt_coast2_est,
            target_pos=list(p_cal_arr),
            mass_init=m_start,
            thrust=thrust, isp=isp,
            impulse_vector=dv_est, # Seed
            tol_km=10.0
        )
        
        # Execute
        mcc_burn_log = jax_planner.evaluate_burn(
             r_start=r_start, v_start=v_start,
             t_start_iso=t_mcc_start_iso,
             dt_seconds=t_burn_mcc,
             lts_params=params_mcc,
             thrust=thrust, isp=isp, mass_init=m_start
        )
        end_mcc = mcc_burn_log[-1]
        
        # --- EVALUATION 2: TCM-1 Accuracy ---
        print("  [Evaluation] Checking TCM-1 Accuracy...")
        t_end_mcc_obj = datetime.fromisoformat(end_mcc['time'].replace('Z', '+00:00'))
        dt_check_2 = (t_arr_obj - t_end_mcc_obj).total_seconds()
        check_2_log = jax_planner.evaluate_trajectory(
            r_start=end_mcc['position'], v_start=end_mcc['velocity'],
            t_start_iso=end_mcc['time'], dt_seconds=dt_check_2,
            mass=end_mcc['mass'], n_steps=100
        )
        r_check_2 = np.array(check_2_log[-1]['position'])
        err_2 = np.linalg.norm(r_check_2 - np.array(p_cal_arr))
        print(f"  TCM-1 Residual Error: {err_2:.1f} km")
        
        if err_2 < 10.0:
            print("  [Step] TCM-1 Accuracy Sufficient. Skipping TCM-2.")
            skip_tcm2 = True
        
    except Exception as e:
        print(f"MCC Failed: {e}")
        skip_tcm2 = True
        full_log = parking_log + burn_log + coast1_log
        last_state = coast1_log[-1]

    # 7b. TCM-2 Setup
    if skip_tcm2:
         if 'end_mcc' in locals():
             t_end_mcc_obj = datetime.fromisoformat(end_mcc['time'].replace('Z', '+00:00'))
             dt_coast2 = (t_arr_obj - t_end_mcc_obj).total_seconds()
             
             coast2_log = jax_planner.evaluate_trajectory(
                r_start=end_mcc['position'], v_start=end_mcc['velocity'],
                t_start_iso=end_mcc['time'], dt_seconds=dt_coast2,
                mass=end_mcc['mass'], n_steps=100
             )
             full_log = parking_log + burn_log + coast1_log + mcc_burn_log + coast2_log
             last_state = coast2_log[-1]
    else:
        # TCM-2 Strategy: 90% of flight time
        t_tcm2_obj = launch_dt_obj + timedelta(seconds=dt_flight_sec * 0.9)
        
        # Propagate Coast 2 to TCM-2 Center (Guess)
        t_mcc_end_obj = datetime.fromisoformat(end_mcc['time'].replace('Z', '+00:00'))
        dt_coast2_guess = (t_tcm2_obj - t_mcc_end_obj).total_seconds()
        
        coast2_log = jax_planner.evaluate_trajectory(
            r_start=end_mcc['position'], v_start=end_mcc['velocity'],
            t_start_iso=end_mcc['time'], dt_seconds=dt_coast2_guess,
            mass=end_mcc['mass'], n_steps=100
        )
        state_tcm2 = coast2_log[-1]
        
        try:
            r_tcm2 = np.array(state_tcm2['position'])
            v_tcm2 = np.array(state_tcm2['velocity'])
            m_tcm2 = state_tcm2['mass']
            
            # Solve Lambert Estimate
            dt_rem_2 = (t_arr_obj - t_tcm2_obj).total_seconds()
            v_mcc_dep, _ = solve_lambert(r_tcm2, np.array(p_cal_arr), dt_rem_2, mu_jup)
            dv_est_2 = v_mcc_dep - v_tcm2
            dv_mag_est_2 = np.linalg.norm(dv_est_2)
            print(f"  TCM-2 Est Delta-V: {dv_mag_est_2*1000:.2f} m/s")
            
            # Estimate Burn Time
            m_dot = thrust / (ve * 1000.0)
            t_burn_est_2 = (m_tcm2 * (1.0 - np.exp(-dv_mag_est_2/ve))) / m_dot
            if t_burn_est_2 < 1.0: t_burn_est_2 = 1.0
            
            # --- HEURISTIC: Half-Burn Offset & Re-Targeting (N-Body) ---
            print("  [Heuristic] Applying Half-Burn Offset & N-Body Optimization for TCM-2...")
            t_tcm2_start_obj = t_tcm2_obj - timedelta(seconds=t_burn_est_2/2.0)
            t_tcm2_start_iso = t_tcm2_start_obj.isoformat().replace('+00:00', 'Z')
            
            # Re-Propagate Coast 2 to start time
            dt_coast2_rev = (t_tcm2_start_obj - t_mcc_end_obj).total_seconds()
            
            coast2_log = jax_planner.evaluate_trajectory(
                r_start=end_mcc['position'], v_start=end_mcc['velocity'],
                t_start_iso=end_mcc['time'], dt_seconds=dt_coast2_rev,
                mass=end_mcc['mass'], n_steps=100
            )
            state_tcm2_start = coast2_log[-1]
            r_start_2 = list(state_tcm2_start['position'])
            v_start_2 = list(state_tcm2_start['velocity'])
            m_start_2 = state_tcm2_start['mass']
            
            # Optimze TCM-2
            dt_coast3_est = (t_arr_obj - (t_tcm2_start_obj + timedelta(seconds=t_burn_est_2))).total_seconds()
            
            params_tcm2 = jax_planner.solve_finite_burn_coast(
                r_start=r_start_2, v_start=v_start_2,
                t_start_iso=t_tcm2_start_iso,
                t_burn_seconds=t_burn_est_2,
                t_coast_seconds=dt_coast3_est,
                target_pos=list(p_cal_arr),
                mass_init=m_start_2,
                thrust=thrust, isp=isp,
                impulse_vector=dv_est_2,
                tol_km=1.0
            )
            
            tcm2_burn_log = jax_planner.evaluate_burn(
                r_start=r_start_2, v_start=v_start_2,
                t_start_iso=t_tcm2_start_iso,
                dt_seconds=t_burn_est_2,
                lts_params=params_tcm2,
                thrust=thrust, isp=isp, mass_init=m_start_2
            )
            
            # Final Coast
            end_tcm2 = tcm2_burn_log[-1]
            dt_coast3 = (t_arr_obj - (t_tcm2_start_obj + timedelta(seconds=t_burn_est_2))).total_seconds()
            
            coast3_log = jax_planner.evaluate_trajectory(
                r_start=end_tcm2['position'], v_start=end_tcm2['velocity'],
                t_start_iso=end_tcm2['time'], dt_seconds=dt_coast3,
                mass=end_tcm2['mass'], n_steps=50
            )
            
            full_log = parking_log + burn_log + coast1_log + mcc_burn_log + coast2_log + tcm2_burn_log + coast3_log
            last_state = coast3_log[-1]
            
            # --- EVALUATION 3: TCM-2 Accuracy (Final) ---
            print("  [Evaluation] Checking TCM-2 Accuracy...")
            err_final_calc = np.linalg.norm(np.array(last_state['position']) - np.array(p_cal_arr))
            print(f"  TCM-2 Residual Error: {err_final_calc:.1f} km")
            
        except Exception as e:
             print(f"TCM-2 Failed: {e}")
             try: full_log = parking_log + burn_log + coast1_log + mcc_burn_log + coast2_log
             except: full_log = parking_log + burn_log
             last_state = coast2_log[-1]

    # Analysis
    r_end = np.array(last_state['position'])
    err = np.linalg.norm(r_end - np.array(p_cal_arr))
    
    print(f"[Result] Final Error: {err:.1f} km")
    
    # Plotting
    # Need to handle empty parking log for plotting?
    r_full = np.array([p['position'] for p in full_log])
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_full[:,0], r_full[:,1], r_full[:,2], label='Trajectory', color='cyan')
    ax.scatter(p_gan[0], p_gan[1], p_gan[2], color='orange', label='Ganymede', s=50)
    ax.scatter(p_cal[0], p_cal[1], p_cal[2], color='gray', label='Callisto', s=50)
    
    ax.set_title(f"JAX LTS Mission: Wait {wait_sec/86400:.1f}d + Fly (Err {err:.1f}km)")
    plt.savefig("scenario_jax_lts.png")
    print("Saved scenario_jax_lts.png")
    
    # Export for Viewer
    json_path = "trajectory_lts.json"
    print(f"Exporting (MissionManifest format) to {json_path}...")
    
    start_iso = full_log[0]['time'] # First log entry (could be parking start)
    start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
    
    formatted_timeline = []
    bodies_to_export = ["ganymede", "callisto", "jupiter"]
    
    for entry in full_log:
        curr_iso = entry['time']
        curr_dt = datetime.fromisoformat(curr_iso.replace('Z', '+00:00'))
        dt_seconds = (curr_dt - start_dt).total_seconds()
        
        bodies_pos = {}
        for b in bodies_to_export:
            if b == 'jupiter':
                bodies_pos[b] = [0.0, 0.0, 0.0]
            else:
                p_b, _ = engine.get_body_state(b, curr_iso) 
                bodies_pos[b] = list(p_b)
        
        formatted_timeline.append({
            "time": dt_seconds,
            "position": entry['position'],
            "velocity": entry['velocity'],
            "mass": entry['mass'],
            "bodies": bodies_pos
        })
        
    manifest = {
        "meta": {
            "missionName": "JAX LTS: Ganymede to Callisto",
            "startTime": start_iso,
            "endTime": full_log[-1]['time'],
            "bodies": ["ganymede", "callisto", "jupiter"]
        },
        "timeline": formatted_timeline,
        "maneuvers": [] 
    }
    
    # Add Maneuver Info (Burn)
    # Burn starts at t_launch.
    # t_launch is relative to start_dt?
    # If parking existed, t_launch > start_dt.
    # start_dt is mission_start.
    burn_start_sec = (launch_dt_obj - start_dt).total_seconds()
    
    manifest["maneuvers"].append({
        "startTime": burn_start_sec,
        "duration": t_burn,
        "deltaV": [0.0, 0.0, 0.0], 
        "type": "finite"
    })

    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    run_jax_lts_scenario()
