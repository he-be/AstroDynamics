
import sys
import os
import time
import numpy as np
from datetime import datetime
from skyfield.api import Loader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from universe.engine import PhysicsEngine
from universe.planning import refine_finite_transfer, refine_lts_transfer, solve_lambert

def run_benchmark():
    print("=== Linear Tangent Steering (LTS) vs Constant Steering Benchmark ===")
    
    # 1. Init
    engine = PhysicsEngine(include_saturn=False)
    load = Loader('data')
    ts = load.timescale()
    
    t_launch = ts.utc(2025, 3, 10, 12, 0, 0)
    t_arrival = ts.utc(2025, 3, 14, 12, 0, 0) # 4 days
    
    # 2. State
    p1, v1 = engine.get_body_state('ganymede', t_launch.utc_iso())
    p2, v2 = engine.get_body_state('callisto', t_arrival.utc_iso())
    pj, vj = engine.get_body_state('jupiter', t_launch.utc_iso())
    pj2, vj2 = engine.get_body_state('jupiter', t_arrival.utc_iso())
    
    r1_jup = np.array(p1) - np.array(pj)
    v1_jup = np.array(v1) - np.array(vj)
    r2_jup = np.array(p2) - np.array(pj2)
    
    mass = 2000.0
    mass = 2000.0
    thrust = 10.0 # 10 N
    isp = 3000.0 # Ion
    
    dt_sec = (t_arrival.utc_datetime() - t_launch.utc_datetime()).total_seconds()
    mu_jup = 1.26686534e8 
    v_dep_lamb, _ = solve_lambert(r1_jup, r2_jup, dt_sec, mu_jup)
    dv_imp = v_dep_lamb - v1_jup
    print(f"Lambert DV Guess: {np.linalg.norm(dv_imp)*1000:.1f} m/s")
    
    # Warmup / JIT
    print("\n[Warmup] Compiling Kernels...")
    # Warmup / JIT
    print("\n[Warmup] Compiling Kernels...")
    # try:
    #     # Constant Kernel
    #     engine.propagate_controlled(list(r1_jup)+list(v1_jup), t_launch.utc_iso(), 0.1, [100.0,0.0,0.0], mass, isp, steering_mode='constant')
    #     # LTS Kernel
    #     ctrl = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     engine.propagate_controlled(list(r1_jup)+list(v1_jup), t_launch.utc_iso(), 0.1, ctrl, mass, isp, steering_mode='linear_tangent', thrust_magnitude=thrust)
    # except Exception as e:
    #     print(f"Prop Error: {e}")
        
    print("Compilation Done.\n")

    # --- Method A: Constant Steering (FD) ---
    print("--- Method A: Constant Steering (Finite Diff) ---")
    t0 = time.time()
    # Note: explicit use_variational=False to force FD
    common_args = {
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
    
    res_const = refine_finite_transfer(**common_args, use_variational=False)
    t_const = time.time() - t0
    print(f"Constant Duration: {t_const:.2f}s")
    
    # --- Method B: Linear Tangent Steering (Analytic Gradient) ---
    print("\n--- Method B: Linear Tangent Steering (Analytic Gradient) ---")
    t0 = time.time()
    res_lts_tuple = refine_lts_transfer(**common_args)
    t_lts = time.time() - t0
    print(f"LTS Duration: {t_lts:.2f}s")
    
    if res_lts_tuple:
        res_lts, final_pos, final_mass = res_lts_tuple
        # Analyze LTS results
        a_vec = res_lts[0:3]
        b_vec = res_lts[3:6]
        print(f"LTS Result: a={a_vec}, b={b_vec}")
        print(f"LTS Final Mass: {final_mass:.2f} kg")
    else:
        print("LTS Failed.")
    print(f"Constant Result: {res_const}")


if __name__ == "__main__":
    run_benchmark()
