import time
import numpy as np
from engine import PhysicsEngine

def run_verification():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    start_time = "2025-01-01T00:00:00Z"
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    y0 = np.concatenate([ship_pos, ship_vel]).tolist()
    
    # Scenarios to test
    # (Name, Duration in Seconds, Cache Step in Seconds)
    scenarios = [
        ("Tactical (48h)", 48 * 3600, 300),
        ("Strategic (30d)", 30 * 24 * 3600, 600),
    ]
    
    print(f"{'Scenario':<20} | {'Method':<10} | {'Time (s)':<10} | {'Pos Error (km)':<15} | {'Vel Error (m/s)':<15}")
    print("-" * 90)
    
    for name, duration, cache_step in scenarios:
        # 1. Exact Run
        t0 = time.time()
        res_exact = engine.propagate(y0, start_time, duration)
        t_exact = time.time() - t0
        
        print(f"{name:<20} | {'Exact':<10} | {t_exact:<10.4f} | {'-':<15} | {'-':<15}")
        
        pos_exact = np.array(res_exact[:3])
        vel_exact = np.array(res_exact[3:])
        
        # 2. Optimized Run (SciPy Interpolated)
        t0 = time.time()
        res_opt = engine.propagate_interpolated(y0, start_time, duration, cache_step=cache_step)
        t_opt = time.time() - t0
        
        pos_opt = np.array(res_opt[:3])
        vel_opt = np.array(res_opt[3:])
        
        err_pos_opt = np.linalg.norm(pos_exact - pos_opt)
        err_vel_opt = np.linalg.norm(vel_exact - vel_opt) * 1000
        
        print(f"{name:<20} | {'SciPy Opt':<10} | {t_opt:<10.4f} | {err_pos_opt:<15.6f} | {err_vel_opt:<15.6f}")
        
        # 3. JIT Run (Numba RK4)
        # Warmup
        if name == "Tactical (48h)":
            engine.propagate_jit(y0, start_time, duration, dt=10.0, cache_step=cache_step)
            
        t0 = time.time()
        res_jit = engine.propagate_jit(y0, start_time, duration, dt=10.0, cache_step=cache_step)
        t_jit = time.time() - t0
        
        pos_jit = np.array(res_jit[:3])
        vel_jit = np.array(res_jit[3:])
        
        err_pos_jit = np.linalg.norm(pos_exact - pos_jit)
        err_vel_jit = np.linalg.norm(vel_exact - vel_jit) * 1000
        
        print(f"{name:<20} | {'Numba JIT':<10} | {t_jit:<10.4f} | {err_pos_jit:<15.6f} | {err_vel_jit:<15.6f}")
        print("-" * 90)

if __name__ == "__main__":
    run_verification()
