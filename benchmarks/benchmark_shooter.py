
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import Loader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from universe.engine import PhysicsEngine
from universe.planning import refine_finite_transfer, solve_lambert

def run_benchmark():
    print("=== Finite Burn Shooter Benchmark: Analytical vs Finite Difference ===")
    print("Configuration: Saturn EXCLUDED (N=43 States)")
    
    # 1. Initialize Engine (Configuration for Speed)
    # Using include_saturn=False to reduce compilation time for benchmark
    engine = PhysicsEngine(include_saturn=False)
    
    # Load Ephemeris (Loader handles caching, Engine does internal loading too)
    load = Loader('data')
    ts = load.timescale()
    
    # 2. Scenario Setup (Ganymede -> Callisto)
    # Using a 4-day transfer window in March 2025
    t_launch = ts.utc(2025, 3, 10, 12, 0, 0)
    t_arrival = ts.utc(2025, 3, 14, 12, 0, 0) # 4.0 days
    
    print(f"\n[Setup] Scenario: Ganymede -> Callisto")
    print(f"Launch: {t_launch.utc_iso()}")
    print(f"Arrival: {t_arrival.utc_iso()}")

    # Initial State
    p1, v1 = engine.get_body_state('ganymede', t_launch.utc_iso())
    p2, v2 = engine.get_body_state('callisto', t_arrival.utc_iso())
    
    # Jupiter State
    pj, vj = engine.get_body_state('jupiter', t_launch.utc_iso())
    pj2, vj2 = engine.get_body_state('jupiter', t_arrival.utc_iso())
    
    r1_jup = np.array(p1) - np.array(pj)
    v1_jup = np.array(v1) - np.array(vj)
    r2_jup = np.array(p2) - np.array(pj2)
    
    # Specs
    mass = 2000.0
    thrust = 1000.0 # 1 kN
    isp = 320.0
    
    # Lambert Guess
    print("Calculating Lambert Initial Guess...")
    dt_sec = (t_arrival.utc_datetime() - t_launch.utc_datetime()).total_seconds()
    mu_jup = 1.26686534e8 
    
    v_dep_lamb, _ = solve_lambert(r1_jup, r2_jup, dt_sec, mu_jup)
    dv_imp = v_dep_lamb - v1_jup
    print(f"Lambert DV Guess: {np.linalg.norm(dv_imp)*1000:.1f} m/s")
    
    inputs = {
        'engine': engine,
        'r_start': r1_jup,
        'v_start': v1_jup,
        't_start': t_launch.utc_iso(),
        't_end': t_arrival.utc_iso(),
        'target_pos_at_end': list(r2_jup),
        'mass': mass,
        'thrust': thrust,
        'isp': isp,
        'seed_dv_vec': dv_imp
    }

    # --- Warmup / Compilation Phase ---
    print("\n[Phase 0] JIT Compilation / Warmup")
    print("Compiling Variational Equations (this generates and compiles C++ code)...")
    t0 = time.time()
    
    # Dummy propagation to trigger JIT
    try:
        dummy_state = list(r1_jup) + list(v1_jup)
        # 0.1s burn
        engine.propagate_controlled(
            dummy_state, t_launch.utc_iso(), 0.1, [100.0, 0.0, 0.0], mass, isp,
            with_variational_equations=True
        )
    except Exception as e:
        print(f"Warmup Error: {e}")
        
    t_compile = time.time() - t0
    print(f"Compilation Finished. Time: {t_compile:.2f} seconds")

    # --- Method A: Analytical (Variational Equations) ---
    print("\n--- Running Method A: Variational Equations (Analytical) ---")
    start_time = time.time()
    
    dv_sol_a = refine_finite_transfer(
        **inputs,
        use_variational=True
    )
    
    end_time = time.time()
    time_a = end_time - start_time
    print(f"Method A Finished in {time_a:.4f} seconds")
    
    # --- Method B: Finite Difference ---
    print("\n--- Running Method B: Finite Difference (Numerical) ---")
    start_time = time.time()
    
    dv_sol_b = refine_finite_transfer(
        **inputs,
        use_variational=False
    )
    
    end_time = time.time()
    time_b = end_time - start_time
    print(f"Method B Finished in {time_b:.4f} seconds")
    
    # --- Results ---
    print("\n=== Benchmark Results ===")
    print(f"{'Metric':<30} | {'Method A (Analytic)':<20} | {'Method B (Finite Diff)':<20}")
    print("-" * 80)
    print(f"{'Execution Time (s)':<30} | {time_a:<20.4f} | {time_b:<20.4f}")
    if time_a > 0:
        print(f"{'Speedup Factor':<30} | {time_b/time_a:<20.1f}x | 1.0x")
    
    print("-" * 80)
    print(f"Initial Compilation Time: {t_compile:.2f} s")
    print("(This is a one-time cost upon first execution)")
    
    diff = np.linalg.norm(dv_sol_a - dv_sol_b) * 1000.0
    print(f"\nResult Discrepancy: {diff:.2f} m/s")

if __name__ == "__main__":
    run_benchmark()
