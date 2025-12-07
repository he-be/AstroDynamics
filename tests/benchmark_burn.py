import sys
import os
import time
import numpy as np

# Add universe to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine

def benchmark_cost():
    print("=== Cost Benchmark: Coast vs Finite Burn ===")
    engine = PhysicsEngine()
    
    # Setup
    t0 = "2025-01-01T00:00:00Z"
    p, v = engine.get_body_state('ganymede', t0)
    state_0 = list(p) + list(v)
    mass = 1000.0
    
    # Duration: 30 minutes (Typical Burn)
    duration = 1800.0 
    
    ITERATIONS = 20
    
    # 1. Coast Benchmark
    print(f"\n[Coast] Running {ITERATIONS} propagations of {duration}s...", flush=True)
    start_time = time.time()
    for _ in range(ITERATIONS):
        engine.propagate(state_0, t0, duration)
    end_time = time.time()
    avg_coast = (end_time - start_time) / ITERATIONS
    print(f"  Avg Coast Time: {avg_coast*1000:.4f} ms")
    
    # 2. Finite Burn Benchmark
    print(f"\n[Burn] Running {ITERATIONS} controlled propagations of {duration}s...", flush=True)
    thrust_vector = [1000.0, 0.0, 0.0] # Inertial thrust
    start_time = time.time()
    for _ in range(ITERATIONS):
        # engine.propagate_controlled(state_0, t0, duration, thrust_vector, mass, isp=3000)
        # Note: We need to use the internal method or ensure Mission primitives aren't adding overhead.
        # Direct engine call is best.
        engine.propagate_controlled(state_0, t0, duration, thrust_vector, mass, 3000.0)
    end_time = time.time()
    avg_burn = (end_time - start_time) / ITERATIONS
    print(f"  Avg Burn Time:  {avg_burn*1000:.4f} ms")
    
    ratio = avg_burn / avg_coast
    print(f"\n[Result] Burn is {ratio:.1f}x times slower than Coast.")
    
    # Estimate Solver Cost
    # Solver typically runs ~15 iterations.
    # Each iteration: 1 Burn (Departure) + 1 Coast (Transit ~6 days).
    # Wait, the "Shooter" for finite burn needs to integrate:
    #   A. Finite Burn Phase (0 to t_burn)
    #   B. Coast Phase (t_burn to t_arrival)
    
    # Let's benchmark "Long Coast" too (6 days)
    duration_long = 6.0 * 86400.0
    print(f"\n[Long Coast] Running {ITERATIONS} propagations of {duration_long/86400:.1f} days...", flush=True)
    start_time = time.time()
    for _ in range(ITERATIONS):
        engine.propagate(state_0, t0, duration_long)
    end_time = time.time()
    avg_long_coast = (end_time - start_time) / ITERATIONS
    print(f"  Avg Long Coast: {avg_long_coast*1000:.4f} ms")
    
    total_iter_cost = avg_burn + avg_long_coast
    print(f"\n[Estimation] One Solver Iteration (Burn + Coast): {total_iter_cost*1000:.2f} ms")
    print(f"[Estimation] Total Convergence (20 iters): {total_iter_cost * 20 * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark_cost()
