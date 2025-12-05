import time
import numpy as np
from engine import PhysicsEngine

def test_caching():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    # Define a simple state (Jovicentric)
    state_0 = [1.06e6, 0, 0, 0, 13.0, 0] # Near Ganymede
    t_start = "2025-01-01T00:00:00Z"
    duration = 3600 # 1 hour
    
    print("\n--- Test 1: Cold Start (Compilation) ---")
    start_time = time.time()
    res1 = engine.propagate(state_0, t_start, duration)
    t1 = time.time() - start_time
    print(f"Cold Start Time: {t1:.4f} seconds")
    print(f"Result 1: {res1[:3]}") # Just print position
    
    # Change state slightly to verify reuse with new initial conditions
    state_1 = [1.07e6, 0, 0, 0, 13.1, 0]
    
    print("\n--- Test 2: Warm Start (Cached Integrator) ---")
    start_time = time.time()
    res2 = engine.propagate(state_1, t_start, duration)
    t2 = time.time() - start_time
    print(f"Warm Start Time: {t2:.4f} seconds")
    print(f"Result 2: {res2[:3]}")
    
    speedup = t1 / t2 if t2 > 0 else 0
    print(f"\nSpeedup Factor: {speedup:.1f}x")
    
    if t2 < 1.0:
        print("SUCCESS: Caching is working (Warm start < 1s).")
    else:
        print("WARNING: Warm start is slow. Caching might not be effective.")
        
    print("\n--- Test 3: Loop Performance (100 iterations) ---")
    start_time = time.time()
    for i in range(100):
        # Vary state slightly
        s = [1.06e6 + i*100, 0, 0, 0, 13.0, 0]
        engine.propagate(s, t_start, duration)
    total_loop = time.time() - start_time
    avg_loop = total_loop / 100
    print(f"Total Loop Time: {total_loop:.4f} sec")
    print(f"Average per call: {avg_loop:.4f} sec")

if __name__ == "__main__":
    test_caching()
