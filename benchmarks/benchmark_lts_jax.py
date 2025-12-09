import time
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import LinearInterpolation
import optax
import equinox as eqx
from datetime import datetime

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from universe.engine import PhysicsEngine
from universe.jax_engine import JAXEngine
from skyfield.api import Loader

def test_benchmark_lts_jax():
    print("=== Linear Tangent Steering (LTS) Optimization with JAX ===")
    
    # 1. Setup Standard Engine (for Ephemeris)
    print("Initializing Skyfield...")
    engine_sky = PhysicsEngine(include_saturn=False)
    load = Loader('data')
    ts = load.timescale()
    
    t_launch = ts.utc(2025, 3, 10, 12, 0, 0)
    t_arrival = ts.utc(2025, 3, 14, 12, 0, 0)
    
    dt_sec = (t_arrival.tt - t_launch.tt) * 86400.0
    
    t_start_date = t_launch.utc_datetime() # For reference if needed
    
    # 2. Prepare Ephemeris Interpolation for JAX
    print("Preparing Ephemeris Interpolator...")
    # Nodes every 1 hour
    t_nodes = np.arange(0, dt_sec + 3600, 3600) 
    
    # We need positions of [Sun, Jup, Sat, Io, Eur, Gan, Cal] relative to JUPITER
    # Note: Jup relative to Jup is 0.
    # Order must match JAXEngine: 0:Sun, 1:Jup, 2:Sat, 3:Io, 4:Eur, 5:Gan, 6:Cal
    # Actually JAXEngine loops over indices [0, 2, 3, 4, 5, 6] for gravity bodies.
    # The interpolator must return shape (6, 3) corresponding to these bodies?
    # No, JAXEngine.get_vector_field docstring says:
    # "Mapping: 0->Sun, 1->Sat, 2->Io, 3->Eur, 4->Gan, 5->Cal" relative to Loop Index i.
    # Wait, let's check jax_engine.py logic.
    # gm_indices = [0, 2, 3, 4, 5, 6]
    # others_pos[i] corresponds to gm_indices[i].
    # So the interpolator should return array of shape (6, 3) where:
    # 0: Sun, 1: Saturn, 2: Io, 3: Europa, 4: Ganymede, 5: Callisto.
    
    body_names = ['sun', 'saturn', 'io', 'europa', 'ganymede', 'callisto']
    
    # Pre-allocate array: (Time, Body, Coords) -> (N_nodes, 6, 3)
    ephem_data = np.zeros((len(t_nodes), 6, 3))
    
    # Optimization: We can batch query Skyfield? Or just loop. Loop is fine for setup.
    # But engine.get_body_state works on single time.
    # Actually we can pass array of times to Skyfield.
    
    # Vectorized Time Generation (TT scale is linear)
    times_sky = ts.tt_jd(t_launch.tt + t_nodes / 86400.0)
    
    # We need to access kernels directly to support array times efficiently, 
    # OR just loop if `engine` doesn't support vector times. 
    # `PhysicsEngine` creates `self.jup_moons`.
    
    # Let's just use a loop for safety and simplicity in this benchmark script.
    # It takes a few seconds but only happens once.
    print(f"Sampling {len(t_nodes)} ephemeris points...")
    
    # To speed up, we can use the kernel directly if available.
    # engine.jup_moons is a dict of body objects.
    jup = engine_sky.planets['jupiter barycenter']
    
    for i, name in enumerate(body_names):
        # We need simpler access.
        # engine.get_body_state handles "relative to Jupiter Center".
        # Let's trust it.
        pass # Logic below
        
    # Vector implementation for speed
    # We'll just define a helper that uses the kernel directly if possible, or loop.
    # Let's loop.
    
    for ti, t_val in enumerate(t_nodes):
        current_iso = times_sky[ti].utc_iso()
        for bi, b_name in enumerate(body_names):
            pos, _ = engine_sky.get_body_state(b_name, current_iso)
            ephem_data[ti, bi, :] = pos
            
    # Create JAX Interpolation
    # Shape: (Time, Data) -> JAX LinearInterpolation expects `ys` with leading time dimension?
    # Yes. `ts` is time nodes. `ys` is data values.
    # Flatten last dimensions? JAXEngine expects `body_pos_flat.reshape(6, 3)`.
    # So `ys` should be (N_t, 6*3) or (N_t, 6, 3) if Diffrax supports structured interp.
    # Diffrax supports PyTrees. So (N_t, 6, 3) is fine.
    
    ephem_jax_data = jnp.array(ephem_data)
    t_nodes_jax = jnp.array(t_nodes)
    
    moon_interp = LinearInterpolation(ts=t_nodes_jax, ys=ephem_jax_data)
    
    print("Method: JAX Engine (Diffrax + JIT)")
    jax_engine = JAXEngine()
    
    # Initial State (Ganymede relative to Jup)
    r0, v0 = engine_sky.get_body_state('ganymede', t_launch.utc_iso())
    
    # Offset to avoid singularity (Start "at" Ganymede -> Start at 5000km altitude)
    r0 = np.array(r0) + np.array([5000.0, 0.0, 0.0])
    
    # Add Initial Escape Velocity (Avoid crashing into Moon)
    # Ganymede orbital velocity is not sufficient if we start "static" relative to it.
    # We add an impulsive "Launch" delta-v of 3.0 km/s tangential to avoid immediate crash.
    v0 = np.array(v0) + np.array([0.0, 3.0, 0.0])
    
    mass_init = 2000.0
    y0 = jnp.concatenate([jnp.array(r0), jnp.array(v0), jnp.array([mass_init])])
    
    # Target (Callisto)
    r_target, _ = engine_sky.get_body_state('callisto', t_arrival.utc_iso())
    r_target = jnp.array(r_target)
    
    # Parameters
    thrust = 10.0
    isp = 3000.0
    # LTS Params: [ax, ay, az, bx, by, bz, flow_rate, thrust_mag]
    # Initial Guess: 0 steering
    g0 = 9.80665
    flow_rate = thrust / (isp * g0)
    
    # Params vector: [ax, ay, az, bx, by, bz] -> 6 vars to optimize
    # Fixed params: flow_rate, thrust (passed separately or appended)
    init_params = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Direction only
    
    # 3. Define Loss Function
    @eqx.filter_jit
    def loss_fn(opt_params):
        # Construct full params for engine
        # [ax, ay, az, bx, by, bz, flow_rate, thrust]
        full_params = jnp.concatenate([opt_params, jnp.array([flow_rate, thrust])])
        
        sol = jax_engine.propagate(
            state_init=y0,
            t_span=(0.0, dt_sec),
            control_params=full_params,
            moon_interp=moon_interp,
            steering_mode='linear_tangent'
        )
        
        final_state = sol.ys[-1]
        r_final = final_state[0:3]
        
        # Distance Error Squared
        loss = jnp.sum((r_final - r_target)**2)
        return loss

    # 4. Optimization Loop (Adam)
    print("Compiling Loss Function...")
    t_start = time.time()
    loss_val, grads = jax.value_and_grad(loss_fn)(init_params)
    grads.block_until_ready()
    t_end = time.time()
    print(f"First Compile & Run: {t_end - t_start:.4f} s")
    print(f"Initial Loss (d^2): {loss_val:.2e}, Grad Norm: {jnp.linalg.norm(grads):.2e}")
    
    # Optimizer Init
    optimizer = optax.adam(learning_rate=1e-3) 
    
    opt_state = optimizer.init(init_params)
    params = init_params
    
    print("Starting Optimization (20 Steps)...")
    t_opt_start = time.time()
    
    @eqx.filter_jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    for i in range(20):
        params, opt_state, loss, grads = step(params, opt_state)
        # Verify specific step timing? No, loop is fast.
        if i % 5 == 0:
            dist_err = jnp.sqrt(loss)
            print(f"  Step {i}: Error {dist_err:.1f} km, Grad {jnp.linalg.norm(grads):.1e}")

    t_opt_end = time.time()
    print(f"Optimization Time (20 steps): {t_opt_end - t_opt_start:.4f} s")
    print(f"Steps per second: {20 / (t_opt_end - t_opt_start):.1f}")
    
    if (t_opt_end - t_opt_start) < 2.0:
        print("SUCCESS: Performance is excellent.")
    else:
        print("WARNING: Slow optimization.")

if __name__ == "__main__":
    test_benchmark_lts_jax()
