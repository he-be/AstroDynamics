
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import LinearInterpolation
import optax
import equinox as eqx
from datetime import datetime, timedelta
from skyfield.api import Loader, Timescale

from universe.engine import PhysicsEngine
from universe.jax_engine import JAXEngine

class JAXPlanner:
    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self.jax_engine = JAXEngine()
        self.load = Loader('data')
        self.ts = self.load.timescale()
        
    def prepare_ephemeris(self, t_start_iso: str, dt_seconds: float, nodes: int = 100):
        """
        Prepares JAX-compatible ephemeris interpolation.
        Sample points from t_start to t_start + dt.
        """
        # Parse start time
        dt_obj = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
        t0 = self.ts.from_datetime(dt_obj)
        
        # Generation time nodes
        # Use simple steps
        t_vals = np.linspace(0, dt_seconds, nodes)
        times_sky = self.ts.tt_jd(t0.tt + t_vals / 86400.0)
        
        # Body list must match JAXEngine expectation:
        # 0:Sun, 1:Sat, 2:Io, 3:Eur, 4:Gan, 5:Cal
        body_names = ['sun', 'saturn barycenter', 'io', 'europa', 'ganymede', 'callisto']
        
        ephem_data = np.zeros((nodes, 6, 3))
        
        # Optimization: Loop over bodies, ask Skyfield for vector positions
        # PhysicsEngine usually computes relative to Jupiter Barycenter.
        jup = self.engine.planets['jupiter barycenter']
        
        for i, name in enumerate(body_names):
            # Check moons first (Io, Eur, Gan, Cal)
            # Keys in moons are likely lowercase? engine.py uses .lower().
            if name.lower() in self.engine.moons:
                body = self.engine.moons[name.lower()]
            else:
                # Then planets (Sun, Saturn)
                body = self.engine.planets[name]
            
            # Vector position relative to Jupiter
            # Skyfield 'at(times)' returns vector position object
            pos = jup.at(times_sky).observe(body).position.km
            # pos shape is (3, N) -> Transpose to (N, 3)
            ephem_data[:, i, :] = pos.T
            
        # Create Interpolator
        t_nodes_jax = jnp.array(t_vals)
        ephem_jax_data = jnp.array(ephem_data)
        
        return LinearInterpolation(ts=t_nodes_jax, ys=ephem_jax_data)

    def solve_lts_transfer(self, 
                          r_start: list, 
                          v_start: list, 
                          t_start_iso: str, 
                          dt_seconds: float, 
                          target_pos: list, 
                          mass_init: float,
                          thrust: float, 
                          isp: float,
                          initial_params_guess: np.ndarray = None):
        """
        Optimizes LTS parameters to reach target_pos at t_start + dt.
        """
        
        # 1. Ephemeris
        # Ensure enough nodes for accuracy. 1 node per hour or so.
        n_nodes = max(50, int(dt_seconds / 3600.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, dt_seconds, nodes=n_nodes)
        
        # 2. Setup JAX Data
        y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([mass_init])])
        r_target_jax = jnp.array(target_pos)
        
        g0 = 9.80665
        flow_rate = thrust / (isp * g0)
        
        if initial_params_guess is None:
            init_params = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            init_params = jnp.array(initial_params_guess)
            
        # 3. Define Optimization Step
        
        @eqx.filter_jit
        def loss_fn(opt_params):
            full_params = jnp.concatenate([opt_params, jnp.array([flow_rate, thrust])])
            sol = self.jax_engine.propagate(
                state_init=y0, 
                t_span=(0.0, dt_seconds), 
                control_params=full_params,
                moon_interp=moon_interp,
                steering_mode='linear_tangent',
                max_steps=5000000
            )
            final_state = sol.ys[-1]
            r_final = final_state[0:3]
            return jnp.sum((r_final - r_target_jax)**2)

        optimizer = optax.adam(learning_rate=1e-2) # Start aggressive
        opt_state = optimizer.init(init_params)
        params = init_params
        
        @eqx.filter_jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
            
        # 4. Run Loop
        # Hard limit iterations for safety in this version
        # Or iterate until convergence
        print("[JAXPlanner] Starting optimization...")
        for i in range(100):
            params, opt_state, loss = step(params, opt_state)
            err_km = float(jnp.sqrt(loss))
            
            if i % 10 == 0:
                print(f"  Iter {i}: Error {err_km:.1f} km")
                
            if err_km < 100.0: # Convergence Threshold
                print(f"  Converged at Iter {i}: {err_km:.1f} km")
                break
                
        # 5. Return result in standard format
        # Need final state for mass and pos
        # Use Python propagation once (or evaluate JAX one last time)
        
        # We need final numpy arrays
        final_params = np.array(params)
        
        # Evaluate final state
        # We can't easily call 'evaluate' on JIT function without compiling it again if separated.
        # But we can just use the loss_fn logic or a separate call.
        # Let's run propagation one last time with correct params outside JIT or inside.
        
        # Actually easier to just call jax_engine.propagate again (it's JITted inside the class usually?)
        # jax_engine.propagate is not JITted by default in the class method, unless wrapped.
        # But we want to return standard types.
        
        full_params = jnp.concatenate([params, jnp.array([flow_rate, thrust])])
        sol = self.jax_engine.propagate(y0, (0.0, dt_seconds), full_params, moon_interp, steering_mode='linear_tangent')
        final_y = sol.ys[-1]
        
        final_pos = np.array(final_y[0:3])
        final_mass = float(final_y[6])
        
        return final_params, final_pos, final_mass

    def solve_impulsive_shooting(self, 
                               r_start: list, 
                               t_start_iso: str, 
                               dt_seconds: float, 
                               r_target: list,
                               initial_v_guess: list = None):
        """
        Solves for the initial velocity to reach r_target after dt_seconds using N-Body dynamics (Shooting method).
        Effectively a high-fidelity Lambert solver.
        """
        # 1. Ephemeris
        n_nodes = max(50, int(dt_seconds / 3600.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, dt_seconds, nodes=n_nodes)
        
        # 2. Setup
        r0 = jnp.array(r_start)
        rt = jnp.array(r_target)
        mass_dummy = 1000.0
        
        if initial_v_guess is None:
            v_guess = (rt - r0) / dt_seconds
        else:
            v_guess = jnp.array(initial_v_guess)
            
        # 3. Loss Function
        @eqx.filter_jit
        def shooting_loss(v_in):
            y0 = jnp.concatenate([r0, v_in, jnp.array([mass_dummy])])
            # Coasting (No Thrust)
            # Control params: [0,0,0, 0] (const steering, flow=0)
            zero_ctrl = jnp.zeros(4) 
            
            # Using 'constant' steering with 0 thrust results in ballistic trajectory
            sol = self.jax_engine.propagate(
                state_init=y0,
                t_span=(0.0, dt_seconds),
                control_params=zero_ctrl,
                moon_interp=moon_interp,
                steering_mode='constant',
                max_steps=5000000
            )
            r_final = sol.ys[-1][0:3]
            return jnp.sum((r_final - rt)**2)
            
        # 4. Optimization
        # Use simple Adam
        optimizer = optax.adam(learning_rate=0.01) # Lower LR for stability
        opt_state = optimizer.init(v_guess)
        params = v_guess
        
        @eqx.filter_jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(shooting_loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        print("[JAXPlanner] Solving Impulsive Shooting...")
        for i in range(500):
            params, opt_state, loss = step(params, opt_state)
            err_km = float(jnp.sqrt(loss))
            if err_km < 1.0: # Tight tolerance for shooting
                print(f"  Converged at Iter {i}: {err_km:.3f} km")
                break
            if i % 100 == 0:
                 print(f"  Iter {i}: Error {err_km:.1f} km")
                
        print(f"  Final Error: {err_km:.3f} km")
        return np.array(params)

    def evaluate_trajectory(self, 
                          r_start: list, 
                          v_start: list, 
                          t_start_iso: str, 
                          dt_seconds: float, 
                          mass: float = 1000.0,
                          n_steps: int = 100):
        """
        Propagates the trajectory using JAX Engine and returns sampled points for visualization.
        """
        # 1. Ephemeris
        n_nodes = max(50, int(dt_seconds / 3600.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, dt_seconds, nodes=n_nodes)
        
        # 2. Setup
        y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([mass])])
        from diffrax import SaveAt, ODETerm, Dopri5, PIDController, diffeqsolve
        
        t_eval = jnp.linspace(0, dt_seconds, n_steps)
        control_params = jnp.zeros(6) 
        
        term = ODETerm(self.jax_engine.get_vector_field(moon_interp, 'constant'))
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
        
        sol = diffeqsolve(
            term, solver,
            t0=0.0, t1=dt_seconds,
            dt0=10.0,
            y0=y0,
            args=control_params,
            stepsize_controller=stepsize_controller,
            max_steps=5000000,
            saveat=SaveAt(ts=t_eval)
        )
        
        # 4. Process Output
        ys = sol.ys # Shape (n_steps, 7)
        ts_res = sol.ts
        
        ys_np = np.array(ys)
        ts_np = np.array(ts_res)
        
        result_log = []
        if t_start_iso.endswith('Z'):
             t_start_iso = t_start_iso.replace('Z', '+00:00')
        start_dt = datetime.fromisoformat(t_start_iso)
        
        for i in range(len(ts_np)):
            t_sec = float(ts_np[i])
            curr_time = start_dt + timedelta(seconds=t_sec)
            t_iso = curr_time.isoformat().replace('+00:00', 'Z')
            
            state = ys_np[i, :]
            result_log.append({
                'time': t_iso,
                'position': state[0:3].tolist(),
                'velocity': state[3:6].tolist(),
                'mass': float(state[6])
            })
            
        return result_log

