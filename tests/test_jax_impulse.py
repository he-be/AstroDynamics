
import pytest
import numpy as np
import time
from datetime import datetime

# Local imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner
from universe.planning import solve_lambert

def test_impulsive_shooting_vs_lambert():
    print("=== Testing JAX Impulsive Shooting vs Keplerian Lambert ===")
    
    # 1. Setup
    engine = PhysicsEngine(include_saturn=False)
    planner = JAXPlanner(engine)
    
    # Scenario: Ganymede -> Callisto (Fast Transfer, ~4 days)
    # Using the same dates as benchmark
    ts = engine.ts
    t_launch = ts.utc(2025, 3, 10, 12, 0, 0)
    t_arrival = ts.utc(2025, 3, 14, 12, 0, 0)
    dt_sec = (t_arrival.tt - t_launch.tt) * 86400.0
    
    r_start, v_body_start = engine.get_body_state('ganymede', t_launch.utc_iso())
    r_target, _ = engine.get_body_state('callisto', t_arrival.utc_iso())
    
    # Offset to avoid singularity (Start at 5000km altitude)
    r_start = np.array(r_start) + np.array([5000.0, 0.0, 0.0])
    
    # Start slightly offset from Ganymede to represent a parking orbit or specific departure point
    # but still "at" Ganymede for Lambert purposes usually. 
    # Let's use the exact body positions for Lambert.
    
    print(f"Transfer Time: {dt_sec/86400:.2f} days")
    
    # 2. Solve Keplerian Lambert (Baseline)
    print("\n--- Solver 1: Keplerian Lambert ---")
    mu_jup = engine.GM['jupiter']
    
    # Lambert needs r1, r2, dt. 
    # Use center-to-center for approximation
    v1_lambert, v2_lambert = solve_lambert(np.array(r_start), np.array(r_target), dt_sec, mu_jup)
    
    print(f"Lambert V1: {v1_lambert}")
    
    # Verify Lambert Error in N-Body
    # Propagate using JAX Engine with Lambert V1
    # Note: JAX Engine simulates N-Body, while Lambert assumes 2-Body.
    # We expect some error (thousands of km).
    
    params_lambert = planner.solve_impulsive_shooting(
        r_start=list(r_start),
        t_start_iso=t_launch.utc_iso(),
        dt_seconds=dt_sec,
        r_target=list(r_target),
        initial_v_guess=list(v1_lambert)
    )
    # Wait, solve_impulsive_shooting *optimizes*. We want to *evaluate* first?
    # No, let's just use solve_impulsive_shooting to see if it *improves* the guess.
    
    print("\n--- Solver 2: JAX N-Body Shooting ---")
    v_start_refined = params_lambert # result of optimization starting from Lambert
    
    print(f"Refined V1: {v_start_refined}")
    diff = np.linalg.norm(v_start_refined - v1_lambert)
    print(f"Correction Delta-V: {diff*1000:.1f} m/s")
    
    # Assertions
    # The JAX solver prints error, but we can verify it returns valid number.
    assert np.all(np.isfinite(v_start_refined))
    
    # Check that correction is not zero (meaning N-Body effects are real)
    assert diff > 1e-5, "N-Body shooting should differ from Keplerian check"
    
    print("SUCCESS: JAX Impulsive Shooting verified.")

if __name__ == "__main__":
    test_impulsive_shooting_vs_lambert()
