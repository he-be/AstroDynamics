
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def run_jax_lts_scenario():
    print("=== US-09: Ganymede to Callisto (JAX LTS Finite Burn) ===")
    
    # 1. Init
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    
    # Mission: Short transit to emphasize steering? 
    # Or standard 4 days.
    # Finite burn usually implies we need to steer for a while or compensate for long burn losses.
    # Let's try a 4-day transfer but with a Lower Thrust Engine (to force finite burn effects).
    
    t_launch_iso = "2025-03-10T12:00:00Z"
    dt_days = 4.0
    dt_sec = dt_days * 86400.0
    
    print(f"Launch: {t_launch_iso}")
    print(f"Duration: {dt_days} days")
    
    # 2. State
    # Start offset from Ganymede
    p_gan, v_gan = engine.get_body_state('ganymede', t_launch_iso)
    r_start = np.array(p_gan) + np.array([5000.0, 0.0, 0.0])
    v_start = np.array(v_gan)
    
    # Target: Callisto
    t_arr_obj = datetime.fromisoformat(t_launch_iso.replace('Z', '+00:00')) + timedelta(days=dt_days)
    t_arr_iso = t_arr_obj.isoformat().replace('+00:00', 'Z')
    p_cal, _ = engine.get_body_state('callisto', t_arr_iso)
    
    # 3. Specs (Lower thrust to make it interesting)
    # Normall 2000N -> 20 min burn.
    # Let's try 500 N. -> 80 min burn?
    # Actually, let's keep it standard but use LTS to find precise injection.
    mass = 1000.0
    thrust = 2000.0
    isp = 3000.0 # Ion or High Eff? No, 3000s is very high. 300s is chemical.
    # 3000s is electric. 2000N electric is HUGE.
    # Let's stick to valid params.
    # If 300s (Chemical), 2000N.
    
    # Scenario: We treat the WHOLE transfer as a potential steering problem? 
    # No, usually LTS is for the burn phase only.
    # But JAXPlanner `solve_lts_transfer` optimizes the WHOLE duration `dt`.
    # If `dt` is 4 days, and we thrust for 4 days, that's a Low Thrust Trajectory.
    # If we thrust for 20 mins then coast, that's a Finite Burn + Coast.
    # Current `solve_lts_transfer` applies control parameters over the ENTIRE `dt` interval.
    # So if we run it for 4 days, it implies continuous thrust (or coast if u=0, but LTS implies active).
    
    # DEMO 1: Continuous Thrust Transfer (e.g. Electric Propulsion)
    # Low Thrust: 1 N.
    # Duration: 4 days.
    # Can we reach Callisto?
    # Delta V = (F/m) * t = (1/1000) * (4*86400) = 345 m/s.
    # Gan-Cal requires ~3 km/s. So NO.
    
    # DEMO 2: High Thrust Finite Burn (Chemical) matching the Impulse plan.
    # We need to burn for ~20 mins, then coast.
    # Our `solve_lts_transfer` assumes the optimization window IS the burn window.
    # Strategy:
    #   Step A: Solve Impulse (Instant) -> get Delta V.
    #   Step B: Calculate Burn Duration `t_burn`.
    #   Step C: Optimize LTS for `t_burn` to reach a "Target Interface Point".
    #           Where is that point? It's the state on the impulsive coast arc at `t_burn`.
    #           This is "Finite Burn Matching".
    
    print("\n[Strategy] Finite Burn Matching")
    
    # A. Impulse Solution
    print("  1. Solving Impulse...")
    # Guess from Lambert
    from universe.planning import solve_lambert
    mu_jup = engine.GM['jupiter']
    v_lambert, _ = solve_lambert(np.array(r_start), np.array(p_cal), dt_sec, mu_jup)
    
    v_impulse = jax_planner.solve_impulsive_shooting(
        list(r_start), t_launch_iso, dt_sec, list(p_cal), list(v_lambert)
    )
    dv_impulse = np.array(v_impulse) - v_start
    dv_mag = np.linalg.norm(dv_impulse)
    print(f"     Impulse DV: {dv_mag*1000:.1f} m/s")
    
    # B. Burn Duration
    g0 = 9.80665
    ve = isp * g0 / 1000.0 # km/s
    # Rocket Eq: dv = ve * ln(m0/mf) -> mf = m0 * exp(-dv/ve)
    # dm = F * Isp? No. F = m_dot * ve_m_s.
    # m_dot = F / (ve * 1000).
    m_dot = thrust / (ve * 1000.0)
    
    # B. Burn Duration
    g0 = 9.80665
    ve = isp * g0 / 1000.0 # km/s
    # Rocket Eq: dv = ve * ln(m0/mf) -> mf = m0 * exp(-dv/ve)
    m_dot = thrust / (ve * 1000.0)
    
    t_burn_ideal = (mass * (1.0 - np.exp(-dv_mag/ve))) / m_dot
    
    # Finite Burn Gravity Loss Compensation:
    # A finite burn is less efficient than an impulse. We need more Delta-V.
    # Add 1% safety margin. 5% was causing timing errors.
    t_burn = t_burn_ideal * 1.01
    
    print(f"  2. Burn Duration: {t_burn:.1f} s (Ideal: {t_burn_ideal:.1f} s)")
    
    # C. Target Interface State
    # Where should we be at t_burn?
    # Propagate the IMPULSIVE trajectory for t_burn seconds.
    # This gives us a position/velocity target that leads to Callisto.
    print("  3. Generating Interface Target (from Impulse Coast)...")
    
    # Use JAX Planner's engine to propagate impulse solution for t_burn
    params_coast = np.zeros(8) # Constant zero
    # y0_impulse = [r, v_impulse, m]
    # But wait, impulse implies v_impulse is achieved INSTANTLY at t=0.
    # So we propagate from (r_start, v_impulse) for t_burn.
    # This is the "Ideal Coast" path.
    # Our Finite Burn should match the position and velocity of this path at t_burn?
    # Not necessarily position, but if we catch up to it, we are good.
    # Actually, simpler: Use `solve_lts_transfer` to hit the FINAL target, but the integration time is `dt`.
    # BUT `solve_lts_transfer` assumes thrust is ON for the whole time.
    # We can't use it for "Burn then Coast" unless we modify it to support "Switch time".
    
    # OPTION: Just demonstrate LTS for the BURN phase only?
    # Target: "Where we would be if we did the impulse".
    # If we match (r, v) at cutoff, we are on the transfer orbit.
    
    # Let's do that. Match the state at t_burn on the impulsive arc.
    
    y0_imp = np.concatenate([r_start, v_impulse, [mass]])
    # Re-use JAX evaluate for short duration
    t_eval = np.linspace(0, t_burn, 2)
    # We need a quick propagate helper or just use evaluate
    coast_log = jax_planner.evaluate_trajectory(
        r_start=list(r_start),
        v_start=list(v_impulse),
        t_start_iso=t_launch_iso,
        dt_seconds=t_burn,
        mass=mass, # Impulse doesn't burn mass instantly but we simulate post-burn motion
        n_steps=2
    )
    target_interface = coast_log[-1]
    r_int = target_interface['position']
    v_int = target_interface['velocity'] # We want to match Velocity too!
    
    print(f"     Target Interface Pos: {r_int}")
    
    # D. Solve LTS for Burn Phase (Targeting Final Arrival via Coast)
    print("  4. Optimizing Finite Burn + Coast Steering...")
    # This targets the actual arrival at Callisto!
    
    dt_coast = dt_sec - t_burn
    
    params_lts = jax_planner.solve_finite_burn_coast(
        r_start=list(r_start),
        v_start=list(v_start),
        t_start_iso=t_launch_iso,
        t_burn_seconds=t_burn,
        t_coast_seconds=dt_coast,
        target_pos=list(p_cal), # Target Callisto directly!
        mass_init=mass,
        thrust=thrust,
        isp=isp,
        impulse_vector=dv_impulse # Seed with optimized impulse direction
    )
    
    print("     Burn Optimized.")
    
    # E. Full Propagation (Burn + Coast)
    print("\n[Execution] Simulating Full Mission...")
    
    # 1. Burn Phase
    burn_log = jax_planner.evaluate_burn( 
         r_start=list(r_start),
         v_start=list(v_start),
         t_start_iso=t_launch_iso,
         dt_seconds=t_burn,
         lts_params=params_lts,
         thrust=thrust,
         isp=isp,
         mass_init=mass # Pass mass
    )
    
    # 2. Coast Phase
    # Start from end of burn
    end_burn = burn_log[-1]
    t_start_coast = end_burn['time']
    r_coast_start = end_burn['position']
    v_coast_start = end_burn['velocity']
    m_coast_start = end_burn['mass']
    
    dt_coast = dt_sec - t_burn
    
    coast_log = jax_planner.evaluate_trajectory(
        r_start=r_coast_start,
        v_start=v_coast_start,
        t_start_iso=t_start_coast,
        dt_seconds=dt_coast,
        mass=m_coast_start,
        n_steps=100
    )
    
    # F. Analyze
    final_entry = coast_log[-1]
    r_final = np.array(final_entry['position'])
    err = np.linalg.norm(r_final - np.array(p_cal))
    print(f"\n[Result] Final Error: {err:.1f} km")
    
    # Visualization
    full_log = burn_log + coast_log
    traj = np.array([x['position'] for x in full_log])
    
    plt.figure(figsize=(10, 10))
    plt.plot(traj[:,0], traj[:,1], 'g-', label='Finite Burn + Coast')
    plt.scatter([p_gan[0]], [p_gan[1]], c='gray', label='Ganymede')
    plt.scatter([p_cal[0]], [p_cal[1]], c='red', label='Callisto')
    plt.axis('equal')
    plt.legend()
    plt.savefig("scenario_jax_lts.png")
    print("Saved scenario_jax_lts.png")


if __name__ == "__main__":
    run_jax_lts_scenario()
