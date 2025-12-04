import time
import numpy as np
from engine import PhysicsEngine

def run_benchmark():
    print("Initializing Engine for Benchmark...")
    engine = PhysicsEngine()
    
    # Initial State (Low Ganymede Orbit)
    start_time = "2150-01-01T00:00:00Z"
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    y0 = np.concatenate([ship_pos, ship_vel]).tolist()
    
    scenarios = [
        ("Tactical (48 Hours)", 48 * 3600),
        ("Strategic (30 Days)", 30 * 24 * 3600),
        ("Campaign (6 Months)", 180 * 24 * 3600)
    ]
    
    print(f"{'Scenario':<25} | {'Sim Time':<15} | {'Wall Time':<10} | {'RTF':<10}")
    print("-" * 70)
    
    for name, duration in scenarios:
        t0 = time.time()
        # Propagate
        # Note: We are calling propagate once for the full duration.
        # The integrator (solve_ivp) handles internal steps.
        engine.propagate(y0, start_time, duration)
        t1 = time.time()
        
        wall_time = t1 - t0
        rtf = duration / wall_time if wall_time > 0 else float('inf')
        
        print(f"{name:<25} | {duration/3600:<10.1f} h    | {wall_time:<10.4f} s | {rtf:<10.1e}x")

if __name__ == "__main__":
    run_benchmark()
