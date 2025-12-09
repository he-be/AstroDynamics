
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner
from universe.planning import solve_lambert

def run_robustness_test():
    print("=== JAX LTS Robustness Verification ===")
    
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    
    # Test 5 dates, spaced by 2 days
    base_date = datetime(2025, 3, 10, 12, 0, 0)
    dates = [base_date + timedelta(days=2*i) for i in range(5)]
    
    results = []
    
    for t_start_obj in dates:
        t_start_iso = t_start_obj.isoformat() + "Z"
        print(f"\n--- Testing Launch Date: {t_start_iso} ---")
        
        try:
            # 1. Setup Scenario
            dt_days = 4.0
            dt_sec = dt_days * 86400.0
            
            p_gan, v_gan = engine.get_body_state('ganymede', t_start_iso)
            r_start = np.array(p_gan) + np.array([5000.0, 0.0, 0.0])
            v_start = np.array(v_gan)
            
            t_arr_obj = t_start_obj + timedelta(days=dt_days)
            t_arr_iso = t_arr_obj.isoformat() + "Z"
            p_cal, _ = engine.get_body_state('callisto', t_arr_iso)
            
            mass = 1000.0
            thrust = 2000.0
            isp = 3000.0
            
            # 2. Impulse Guess
            print("  1. Solving Impulse...")
            
            # Use Lambert for guess
            mu_jup = engine.GM['jupiter']
            v_lambert, _ = solve_lambert(np.array(r_start), np.array(p_cal), dt_sec, mu_jup)
            
            v_impulse = jax_planner.solve_impulsive_shooting(
                list(r_start), t_start_iso, dt_sec, list(p_cal), 
                initial_v_guess=list(v_lambert)
            )
            dv_impulse = np.array(v_impulse) - v_start
            dv_mag = np.linalg.norm(dv_impulse)
            print(f"     Impulse DV: {dv_mag*1000:.1f} m/s")
            
            # 3. Burn Sizing and LTS
            g0 = 9.80665
            ve = isp * g0 / 1000.0
            m_dot = thrust / (ve * 1000.0)
            
            t_burn_ideal = (mass * (1.0 - np.exp(-dv_mag/ve))) / m_dot
            t_burn = t_burn_ideal * 1.01 # 1% margin
            
            dt_coast = dt_sec - t_burn
            
            # 4. LTS Optimization
            print("  2. Solving LTS...")
            params_lts = jax_planner.solve_finite_burn_coast(
                r_start=list(r_start),
                v_start=list(v_start),
                t_start_iso=t_start_iso,
                t_burn_seconds=t_burn,
                t_coast_seconds=dt_coast,
                target_pos=list(p_cal),
                mass_init=mass,
                thrust=thrust,
                isp=isp,
                impulse_vector=dv_impulse
            )
            
            # 5. Verify
            burn_log = jax_planner.evaluate_burn( 
                 r_start=list(r_start),
                 v_start=list(v_start),
                 t_start_iso=t_start_iso,
                 dt_seconds=t_burn,
                 lts_params=params_lts,
                 thrust=thrust,
                 isp=isp,
                 mass_init=mass
            )
            end_burn = burn_log[-1]
            
            coast_log = jax_planner.evaluate_trajectory(
                r_start=end_burn['position'],
                v_start=end_burn['velocity'],
                t_start_iso=end_burn['time'],
                dt_seconds=dt_coast,
                mass=end_burn['mass'],
                n_steps=100
            )
            
            final_pos = np.array(coast_log[-1]['position'])
            err = np.linalg.norm(final_pos - np.array(p_cal))
            print(f"  Result: Error {err:.1f} km")
            
            results.append({
                "date": t_start_iso,
                "error": err,
                "dv": dv_mag*1000
            })
            
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "date": t_start_iso,
                "error": -1.0,
                "note": str(e)
            })

    # Report
    print("\n=== Summary ===")
    success_count = 0
    for res in results:
        status = "PASS" if res['error'] > 0 and res['error'] < 2500 else "FAIL"
        if status == "PASS": success_count += 1
        print(f"{res['date']}: Error {res['error']:.1f} km (DV: {res.get('dv',0):.1f} m/s) -> {status}")
        
    print(f"\nSuccess Rate: {success_count}/{len(dates)}")

if __name__ == "__main__":
    run_robustness_test()
