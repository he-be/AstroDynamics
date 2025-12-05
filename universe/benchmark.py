import time
import numpy as np
from engine import PhysicsEngine

def run_benchmark():
    print("Initializing Engine for Benchmark...")
    engine = PhysicsEngine()
    
    # Initial State (Low Ganymede Orbit)
    start_time = "2025-01-01T00:00:00Z"
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    y0 = np.concatenate([ship_pos, ship_vel]).tolist()
    
    scenarios = {
        "Tactical (48 Hours)": 48,
        "Strategic (30 Days)": 30 * 24,
        "Campaign (6 Months)": 180 * 24
    }
    
    print(f"{'Scenario':<25} | {'Sim Time':<15} | {'Wall Time':<10} | {'RTF':<10}")
    print("-" * 70)
    
    for name, duration_hours in scenarios.items():
        duration_sec = duration_hours * 3600
        
        # 1. Interpolated (SciPy)
        start = time.time()
        engine.propagate_interpolated(y0, start_time, duration_sec, cache_step=600)
        end = time.time()
        wall_time_interp = end - start
        rtf_interp = duration_sec / wall_time_interp
        
        # 2. JIT (Numba)
        # First run includes compilation time, so we run twice
        if name == "Tactical (48 Hours)":
            print("Compiling JIT functions (warmup)...")
            engine.propagate_jit(y0, start_time, duration_sec, dt=10.0, cache_step=600)
            
        start = time.time()
        engine.propagate_jit(y0, start_time, duration_sec, dt=10.0, cache_step=600)
        end = time.time()
        wall_time_jit = end - start
        rtf_jit = duration_sec / wall_time_jit
        
        print(f"{name:<25} | {duration_hours:<10} h    | {wall_time_interp:<10.4f} s | {rtf_interp:<10.1e} x (SciPy)")
        print(f"{'':<25} | {'':<10}      | {wall_time_jit:<10.4f} s | {rtf_jit:<10.1e} x (Numba JIT)")
        print("-" * 70)

if __name__ == "__main__":
    run_benchmark()
