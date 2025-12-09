
import pytest
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def test_jax_lts_solver_basic():
    """
    Verifies that solve_lts_transfer can converge on a simple finite burn target.
    """
    print("\n[Test] Initializing JAXPlanner for LTS...")
    engine = PhysicsEngine()
    planner = JAXPlanner(engine)
    
    # 1. Setup Scenario: Short finite burn in deep space (far from gravity for simplicity, or just inclusion)
    # We'll stick to Jupiter frame but start far out or just use standard state.
    # Let's use Ganymede vicinity but short time so gravity is linear-ish.
    
    t_start = "2025-01-01T00:00:00Z"
    r0 = [1e6, 0.0, 0.0] # 1 Million km from Jupiter
    v0 = [0.0, 12.0, 0.0] # Moving y-wards at 12 km/s
    
    # Dynamics: r_final ~= r0 + v0*t + 0.5*a*t^2 (roughly)
    dt = 1000.0 # 1000 seconds
    
    # Target: Assume we thrust X-wards.
    # Thrust accel approx: F/m.
    mass = 1000.0
    thrust = 1000.0 # 1000 N
    isp = 300.0
    g0 = 9.80665
    
    # Approx Accel = 1000 N / 1000 kg = 1 m/s^2 = 0.001 km/s^2.
    # Delta V = 0.001 * 1000 = 1.0 km/s.
    # Position shift due to thrust = 0.5 * 0.001 * 1000^2 = 500 km.
    
    # Let's set a target that requires this thrust.
    # Natural ballistic end:
    # r_ballistic = [1e6, 12000, 0] (approx)
    # Target = Ballistic + [500, 0, 0] (Pure X thrust)
    # We use the engine to get true ballistic point roughly, or just let solver find it.
    
    # But to ensure it's reachable, let's pick a point we know is close to reachable.
    # Let's run a forward pass with the engine to generate a "Truth" target with specific params.
    
    print("  Generating Ground Truth Target...")
    # Params: a=[1,0,0], b=[0,0,0] (Constant X thrust)
    params_truth = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, thrust/(isp*g0), thrust])
    # Need Ephemeris
    moon_interp = planner.prepare_ephemeris(t_start, dt, nodes=10)
    
    # Propagate Truth
    y0_arr = jnp.concatenate([jnp.array(r0), jnp.array(v0), jnp.array([mass])])
    sol = planner.jax_engine.propagate(
        y0_arr, (0.0, dt), params_truth, moon_interp, steering_mode='linear_tangent'
    )
    r_truth = np.array(sol.ys[-1][0:3])
    print(f"  Truth Target: {r_truth}")
    
    # 2. Run Solver to Recover Params (or hitting target)
    print("  Running LTS Solver to hit Truth Target...")
    # Initial guess: Close to truth to verify fine-tuning capability.
    # Truth is [1,0,0...]. Guess [0.9, 0.1, 0...]
    guess = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    params_sol, r_final, m_final = planner.solve_lts_transfer(
        r_start=r0,
        v_start=v0,
        t_start_iso=t_start,
        dt_seconds=dt,
        target_pos=list(r_truth),
        mass_init=mass,
        thrust=thrust,
        isp=isp,
        initial_params_guess=guess
    )
    
    # 3. Assertions
    error = np.linalg.norm(r_final - r_truth)
    print(f"  Result Error: {error:.4f} km")
    
    assert error < 1.0, f"LTS Solver failed to converge. Error: {error} km"
    print("PASS: LTS Solver hit the target.")

if __name__ == "__main__":
    test_jax_lts_solver_basic()
