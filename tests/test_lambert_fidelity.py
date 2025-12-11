
import numpy as np
import jax.numpy as jnp
import os
import sys
from datetime import datetime, timedelta

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner
from universe.planning import solve_lambert

def test_lambert_fidelity():
    print("=== Lambert Solver vs Reality Check ===")
    
    # 1. Setup
    engine = PhysicsEngine()
    jax_planner = JAXPlanner(engine)
    jax_engine = jax_planner.jax_engine
    
    # 2. Scenario (Ganymede -> Callisto)
    t_start_iso = "2025-08-31T16:00:00Z"
    flight_time_days = 6.0
    dt_sec = flight_time_days * 86400.0
    t_end_obj = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00')) + timedelta(seconds=dt_sec)
    t_end_iso = t_end_obj.isoformat().replace('+00:00', 'Z')
    
    # 3. Geometry (Jovicentric)
    p_gan, v_gan = engine.get_body_state('ganymede', t_start_iso)
    p_cal, v_cal = engine.get_body_state('callisto', t_end_iso)
    
    r1 = np.array(p_gan)
    r2 = np.array(p_cal)
    
    print(f"Transfer: {t_start_iso} -> {t_end_iso} ({flight_time_days} days)")
    print(f"R1 (Ganymede): {np.linalg.norm(r1):.1f} km")
    print(f"R2 (Callisto): {np.linalg.norm(r2):.1f} km")
    
    # 4. Lambert Solution
    mu_jup = engine.GM['jupiter']
    print(f"Using Mu_Jupiter: {mu_jup}")
    
    v_dep, v_arr = solve_lambert(r1, r2, dt_sec, mu_jup)
    
    # 5. TEST 1: Pure Kepler Propagation (Verification of Solver)
    # Simple RK4 for 2-body problem
    def kepler_acc(r):
        r_mag = np.linalg.norm(r)
        return -mu_jup * r / (r_mag**3)
        
    r_curr = r1.copy()
    v_curr = v_dep.copy()
    dt_step = 10.0
    n_steps = int(dt_sec / dt_step)
    
    for _ in range(n_steps):
        r = r_curr
        v = v_curr
        
        k1_v = kepler_acc(r)
        k1_r = v
        
        k2_v = kepler_acc(r + 0.5*dt_step*k1_r)
        k2_r = v + 0.5*dt_step*k1_v
        
        k3_v = kepler_acc(r + 0.5*dt_step*k2_r)
        k3_r = v + 0.5*dt_step*k2_v
        
        k4_v = kepler_acc(r + dt_step*k3_r)
        k4_r = v + dt_step*k3_v
        
        v_curr = v + (dt_step/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        r_curr = r + (dt_step/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        
    err_kepler = np.linalg.norm(r_curr - r2)
    print(f"\n[Test 1] Lambert vs Kepler Propagator (2-Body Checks)")
    # Should be essentially 0 (or close, typical Lambert tolerance 1e-5)
    print(f"  Target  : {r2}")
    print(f"  Actual  : {r_curr}")
    print(f"  Error   : {err_kepler:.1f} km")
    if err_kepler < 100.0:
        print("  -> PASS: Lambert solver is mathematically consistent.")
    else:
        print("  -> FAIL: Lambert solver output does not match Kepler propagation!")

    # 6. TEST 2: Full N-Body Propagation (Reality Gap)
    # Using JAX Engine (high fidelity)
    print(f"\n[Test 2] Lambert vs N-Body (JAX Engine)")
    
    # Step 2 hours into the transfer to escape Ganymede's gravity well singularity
    dt_coast = 7200.0 
    
    # Propagate Kepler (2-body) to t0 + dt_coast
    r_start = r1.copy()
    v_start = v_dep.copy()
    
    # Quick RK4
    dt_step = 10.0
    steps = int(dt_coast / dt_step)
    for _ in range(steps):
        r = r_start
        v = v_start
        k1_v = kepler_acc(r)
        k1_r = v
        k2_v = kepler_acc(r + 0.5*dt_step*k1_r)
        k2_r = v + 0.5*dt_step*k1_v
        k3_v = kepler_acc(r + 0.5*dt_step*k2_r)
        k3_r = v + 0.5*dt_step*k2_v
        k4_v = kepler_acc(r + dt_step*k3_r)
        k4_r = v + dt_step*k3_v
        v_start = v + (dt_step/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        r_start = r + (dt_step/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        
    print(f"  Starting N-Body test {dt_coast/3600:.1f} hours after departure (to avoid singularity).")
    print(f"  Start R: {np.linalg.norm(r_start):.1f} km")
    
    # New Time
    t_start_nbody_obj = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00')) + timedelta(seconds=dt_coast)
    t_start_nbody = t_start_nbody_obj.isoformat().replace('+00:00', 'Z')
    dt_rem = dt_sec - dt_coast

    # Prepare Ephem for remaining time
    moon_interp = jax_planner.prepare_ephemeris(t_start_nbody, dt_rem, nodes=500)
    
    dynamics_fn = jax_engine.get_vector_field(moon_interp, steering_mode='constant')
    y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([1000.0])])
    zero_ctrl = jnp.zeros(4)
    
    # Check Accel again
    y_dot = dynamics_fn(0.0, y0, zero_ctrl)
    acc_mag = np.linalg.norm(np.array(y_dot[3:6]))
    print(f"  Initial Acceleration: {acc_mag:.3e} km/s^2")
    
    sol = jax_engine.propagate(
        state_init=y0,
        t_span=(0.0, dt_rem),
        control_params=zero_ctrl,
        moon_interp=moon_interp,
        steering_mode='constant',
        max_steps=1000000
    )
    
    r_nbody = np.array(sol.ys[-1][0:3])
    err_nbody = np.linalg.norm(r_nbody - r2)
    
    print(f"  Target  : {r2}")
    print(f"  Actual  : {r_nbody}")
    print(f"  Error   : {err_nbody:.1f} km")
    
    # Analyze perturbation magnitude
    perturbation = np.linalg.norm(r_nbody - r_curr)
    print(f"  Perturbation (N-Body Effect): {perturbation:.1f} km")
    
if __name__ == "__main__":
    test_lambert_fidelity()
