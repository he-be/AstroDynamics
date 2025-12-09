
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
        Optimizes LTS parameters [ax, ay, az, bx, by, bz] to reach target_pos at t_start + dt.
        Control Law: u(t) = unit(a + b*t)
        """
        
        # 1. Ephemeris
        n_nodes = max(50, int(dt_seconds / 3600.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, dt_seconds, nodes=n_nodes)
        
        # 2. Setup JAX Data
        y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([mass_init])])
        r_target_jax = jnp.array(target_pos)
        
        g0 = 9.80665
        flow_rate = thrust / (isp * g0)
        
        if initial_params_guess is None:
            # Default: specific direction (e.g. velocity direction) and zero rate
            # Best guess: unit vector towards target.
            diff = r_target_jax - y0[0:3]
            u_guess = diff / jnp.linalg.norm(diff)
            init_params = jnp.concatenate([u_guess, jnp.array([0.0, 0.0, 0.0])])
        else:
            init_params = jnp.array(initial_params_guess)
            
        # 3. Define Optimization Step
        # 3. Define Optimization Step
        # Helper to scale params
        def unpack_params(p):
            a = p[0:3]
            b_norm = p[3:6]
            b = b_norm / dt_seconds # Scale b so optimization var is O(1)
            return jnp.concatenate([a, b])
            
        @eqx.filter_jit
        def loss_fn(opt_params):
            # opt_params: [ax, ay, az, bx_norm, by_norm, bz_norm]
            scaled_ab = unpack_params(opt_params)
            
            full_params = jnp.concatenate([scaled_ab, jnp.array([flow_rate, thrust])])
            sol = self.jax_engine.propagate(
                state_init=y0, 
                t_span=(0.0, dt_seconds), 
                control_params=full_params,
                moon_interp=moon_interp,
                steering_mode='linear_tangent',
                rtol=1e-9, atol=1e-9,
                max_steps=5000000
            )
            final_state = sol.ys[-1]
            r_final = final_state[0:3]
            pos_err = jnp.sum((r_final - r_target_jax)**2)
            
            # Regularization: Keep |a| ~ 1.0 to avoid scale drift
            a_vec = scaled_ab[0:3]
            reg_err = (jnp.linalg.norm(a_vec) - 1.0)**2
            
            return pos_err + 1000.0 * reg_err

        # Learning Rate: Decay schedule for fine convergence
        # 0.05 (0-250), 0.005 (250-500), 0.001 (500+)
        schedule = optax.piecewise_constant_schedule(
            init_value=0.05,
            boundaries_and_scales={250: 0.1, 500: 0.1} 
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(init_params)
        params = init_params
        
        @eqx.filter_jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
            
        # 4. Run Loop
        print("[JAXPlanner] Solving LTS Transfer...")
        final_err = 0.0
        
        for i in range(1000):
            params, opt_state, loss = step(params, opt_state)
            err_km = float(jnp.sqrt(loss))
            final_err = err_km
            
            if err_km < 0.1: # Convergence Threshold (Super tight)
                print(f"  Converged at Iter {i}: {err_km:.3f} km")
                break
                
            if i % 100 == 0:
                print(f"  Iter {i}: Error {err_km:.3f} km")
                
        # 5. Return result
        # Must unscale parameters for external use
        scaled_ab = unpack_params(params)
        full_params_out = jnp.concatenate([scaled_ab, jnp.array([flow_rate, thrust])])
        
        # One last propagation for Exact State
        sol = self.jax_engine.propagate(y0, (0.0, dt_seconds), full_params_out, moon_interp, steering_mode='linear_tangent')
        final_y = sol.ys[-1]
        
        final_pos = np.array(final_y[0:3])
        final_mass = float(final_y[6])
        
        print(f"  Final LTS Error: {final_err:.1f} km")
        return np.array(params), final_pos, final_mass

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
        n_nodes = max(500, int(dt_seconds / 120.0))
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
        # 4. Optimization
        # Use simple Adam with Schedule
        schedule = optax.piecewise_constant_schedule(
            init_value=0.1,
            boundaries_and_scales={500: 0.1, 1000: 0.1}
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(v_guess)
        params = v_guess
        
        @eqx.filter_jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(shooting_loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        print("[JAXPlanner] Solving Impulsive Shooting...")
        for i in range(2000):
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
        n_nodes = max(500, int(dt_seconds / 120.0))
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

    def solve_finite_burn_coast(self, 
                              r_start: list, 
                              v_start: list, 
                              t_start_iso: str, 
                              t_burn_seconds: float,
                              t_coast_seconds: float,
                              target_pos: list, 
                              mass_init: float,
                              thrust: float, 
                              isp: float,
                              initial_params_guess: np.ndarray = None,
                              impulse_vector: np.ndarray = None):
        """
        Optimizes LTS parameters for a finite burn followed by a coast phase.
        Global Objective: Hit target_pos at t_start + t_burn + t_coast.
        """
        
        # 1. Ephemeris (Need coverage for burn + coast)
        total_dt = t_burn_seconds + t_coast_seconds
        n_nodes = max(500, int(total_dt / 120.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, total_dt, nodes=n_nodes)
        
        # 2. Setup
        y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([mass_init])])
        r_target_jax = jnp.array(target_pos)
        
        g0 = 9.80665
        flow_rate = thrust / (isp * g0)
        
        if initial_params_guess is None:
             if impulse_vector is not None:
                 # Align 'a' with impulse direction
                 iv = jnp.array(impulse_vector)
                 norm = jnp.linalg.norm(iv)
                 u_guess = jnp.where(norm > 1e-9, iv / norm, jnp.array([1.0, 0.0, 0.0]))
                 init_params = jnp.concatenate([u_guess, jnp.array([0.0, 0.0, 0.0])])
             else:
                 init_params = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
             init_params = jnp.array(initial_params_guess)
             
        # Helper: unpack
        def unpack_params(p):
            a = p[0:3]
            b_norm = p[3:6]
            b = b_norm / t_burn_seconds # Scale by BURN duration
            return jnp.concatenate([a, b])

        # 3. Composable Loss Function (Burn + Coast)
        @eqx.filter_jit
        def loss_fn(opt_params):
            scaled_ab = unpack_params(opt_params)
            
            # Phase 1: Burn
            full_params_burn = jnp.concatenate([scaled_ab, jnp.array([flow_rate, thrust])])
            
            sol_burn = self.jax_engine.propagate(
                state_init=y0, 
                t_span=(0.0, t_burn_seconds), 
                control_params=full_params_burn,
                moon_interp=moon_interp,
                steering_mode='linear_tangent',
                rtol=1e-6, atol=1e-6,
                max_steps=1000000
            )
            state_burn_end = sol_burn.ys[-1]
            
            # Phase 2: Coast
            # Params for 'constant' mode with 0 thrust: [0,0,0, 0]
            zero_convex = jnp.array([0.0, 0.0, 0.0, 0.0])
            
            sol_coast = self.jax_engine.propagate(
                state_init=state_burn_end,
                t_span=(t_burn_seconds, total_dt),
                control_params=zero_convex,
                moon_interp=moon_interp,
                steering_mode='constant',
                rtol=1e-6, atol=1e-6,
                max_steps=1000000
            )
            
            final_state = sol_coast.ys[-1]
            r_final = final_state[0:3]
            
            pos_err = jnp.sum((r_final - r_target_jax)**2)
            
            a_vec = scaled_ab[0:3]
            reg_err = (jnp.linalg.norm(a_vec) - 1.0)**2
            
            return pos_err + 1000.0 * reg_err

        # Optimization Loop
        schedule = optax.piecewise_constant_schedule(
            init_value=0.05,
            boundaries_and_scales={250: 0.1, 500: 0.1}
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(init_params)
        params = init_params
        
        @eqx.filter_jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        print("[JAXPlanner] Solving Finite Burn + Coast...")
        for i in range(3000):
            params, opt_state, loss = step(params, opt_state)
            err_km = float(jnp.sqrt(loss)) 
            
            if i % 50 == 0:
                print(f"  Iter {i}: LossSq {err_km:.1f}")
                
            if err_km < 100.0: 
                 break
        
        return np.array(params)

    def evaluate_burn(self, 
                    r_start: list, 
                    v_start: list, 
                    t_start_iso: str, 
                    dt_seconds: float, 
                    lts_params: np.ndarray,
                    thrust: float,
                    isp: float,
                    mass_init: float = 1000.0,
                    n_steps: int = 50):
        """
        Propagates a finite burn trajectory using JAX Engine.
        """
        n_nodes = max(500, int(dt_seconds / 120.0))
        moon_interp = self.prepare_ephemeris(t_start_iso, dt_seconds, nodes=n_nodes)
        
        y0 = jnp.concatenate([jnp.array(r_start), jnp.array(v_start), jnp.array([mass_init])])
        
        g0 = 9.80665
        flow_rate = thrust / (isp * g0)
        
        # Parameters for Engine: params are [a, b_norm] from optimizer
        p = jnp.array(lts_params)
        a = p[0:3]
        b_norm = p[3:6]
        b = b_norm / dt_seconds
        scaled_ab = jnp.concatenate([a, b])
        
        full_params = jnp.concatenate([scaled_ab, jnp.array([flow_rate, thrust])])
        
        from diffrax import SaveAt, ODETerm, Dopri5, PIDController, diffeqsolve
        
        t_eval = jnp.linspace(0, dt_seconds, n_steps)
        
        term = ODETerm(self.jax_engine.get_vector_field(moon_interp, 'linear_tangent'))
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
        
        sol = diffeqsolve(
            term, solver,
            t0=0.0, t1=dt_seconds,
            dt0=1.0,
            y0=y0,
            args=full_params,
            stepsize_controller=stepsize_controller,
            max_steps=100000,
            saveat=SaveAt(ts=t_eval)
        )
        
        # Process Output
        ys_np = np.array(sol.ys)
        ts_np = np.array(sol.ts)
        
        result_log = []
        if t_start_iso.endswith('Z'): t_start_iso = t_start_iso.replace('Z', '+00:00')
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

