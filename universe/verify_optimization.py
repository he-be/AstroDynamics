import time
import numpy as np
from engine import PhysicsEngine

def run_verification():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    start_time = "2150-01-01T00:00:00Z"
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    y0 = np.concatenate([ship_pos, ship_vel]).tolist()
    
    # Scenarios to test
    # (Name, Duration in Seconds, Cache Step in Seconds)
    scenarios = [
        ("Tactical (48h)", 48 * 3600, 300),
        ("Strategic (30d)", 30 * 24 * 3600, 600), # Finer cache (10 mins) for better accuracy
    ]
    
    print(f"{'Scenario':<20} | {'Method':<10} | {'Time (s)':<10} | {'Pos Error (km)':<15} | {'Vel Error (m/s)':<15}")
    print("-" * 90)
    
    for name, duration, cache_step in scenarios:
        # 1. Exact Run
        t0 = time.time()
        res_exact = engine.propagate(y0, start_time, duration)
        t_exact = time.time() - t0
        
        print(f"{name:<20} | {'Exact':<10} | {t_exact:<10.4f} | {'-':<15} | {'-':<15}")
        
        # 2. Optimized Run
        t0 = time.time()
        res_opt = engine.propagate_interpolated(y0, start_time, duration, cache_step=cache_step)
        t_opt = time.time() - t0
        
        # Calculate Error
        pos_exact = np.array(res_exact[:3])
        vel_exact = np.array(res_exact[3:])
        pos_opt = np.array(res_opt[:3])
        vel_opt = np.array(res_opt[3:])
        
        err_pos = np.linalg.norm(pos_exact - pos_opt)
        err_vel = np.linalg.norm(vel_exact - vel_opt) * 1000 # m/s
        
        print(f"{name:<20} | {'Optimized':<10} | {t_opt:<10.4f} | {err_pos:<15.6f} | {err_vel:<15.6f}")
        print(f"{' ':<20} | Speedup: {t_exact/t_opt:.1f}x")
        print("-" * 90)

if __name__ == "__main__":
    run_verification()
