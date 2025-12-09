
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

def test_jax_planner_ganymede_to_callisto():
    print("=== Testing JAXPlanner Integration ===")
    
    # 1. Setup
    engine = PhysicsEngine(include_saturn=False)
    planner = JAXPlanner(engine)
    
    # Scenario: Ganymede -> Callisto (same as benchmark)
    ts = engine.ts
    t_launch = ts.utc(2025, 3, 10, 12, 0, 0)
    t_arrival = ts.utc(2025, 3, 14, 12, 0, 0)
    dt_sec = (t_arrival.tt - t_launch.tt) * 86400.0
    
    r0, v0 = engine.get_body_state('ganymede', t_launch.utc_iso())
    # Offset & Initial V for stability/realism (as found in benchmark)
    r0 = np.array(r0) + np.array([5000.0, 0.0, 0.0])
    v0 = np.array(v0) + np.array([0.0, 3.0, 0.0])
    
    target_pos, _ = engine.get_body_state('callisto', t_arrival.utc_iso())
    
    # Run Solver
    t0 = time.time()
    params, final_pos, final_mass = planner.solve_lts_transfer(
        r_start=list(r0),
        v_start=list(v0),
        t_start_iso=t_launch.utc_iso(),
        dt_seconds=dt_sec,
        target_pos=list(target_pos),
        mass_init=2000.0,
        thrust=10.0,
        isp=3000.0
    )
    t_end = time.time()
    
    print(f"Solver Time: {t_end - t0:.4f} s")
    print(f"Final Position: {final_pos}")
    print(f"Target Position: {target_pos}")
    
    err = np.linalg.norm(final_pos - np.array(target_pos))
    print(f"Final Error: {err:.2f} km")
    
    # Assertions
    assert err < 1000.0, f"Error too high: {err:.2f} km"
    assert final_mass < 2000.0, "Mass should decrease"
    print("SUCCESS: JAXPlanner integration verified.")

if __name__ == "__main__":
    test_jax_planner_ganymede_to_callisto()
