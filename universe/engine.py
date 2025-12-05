import os
import numpy as np
from skyfield.api import Loader, Topos
from scipy.integrate import solve_ivp

class PhysicsEngine:
    def __init__(self, data_dir='./data'):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.load = Loader(data_dir)
        print("Loading Ephemeris kernels... (this may take a while on first run)")
        # DE440s: Planets (Sun, Jupiter Barycenter)
        self.planets = self.load('de440s.bsp')
        # JUP365: Jupiter Moons
        self.jup_moons = self.load('jup365.bsp')
        
        self.ts = self.load.timescale()
        
        # Bodies
        self.sun = self.planets['sun']
        self.jupiter_bary = self.planets['jupiter barycenter']
        self.jupiter = self.jup_moons['jupiter'] # Center of Jupiter from moon kernel
        
        self.moons = {
            'io': self.jup_moons['io'],
            'europa': self.jup_moons['europa'],
            'ganymede': self.jup_moons['ganymede'],
            'callisto': self.jup_moons['callisto']
        }
        
        # GM values (km^3/s^2) - Approximate or loaded from kernel if possible
        # For high precision we should extract from kernel, but hardcoding standard values is safer for MVP
        self.GM = {
            'sun': 1.32712440018e11,
            'jupiter': 1.26686534e8,
            'io': 5959.91,
            'europa': 3202.73,
            'ganymede': 9887.83,
            'callisto': 7179.28
        }

    def get_body_state(self, body_name: str, time_iso: str):
        """
        Get position (km) and velocity (km/s) of a body relative to JUPITER CENTER (ICRS frame).
        """
        from datetime import datetime, timezone
        # Handle Z for UTC
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        t = self.ts.from_datetime(dt_obj)
        
        if body_name.lower() == 'jupiter':
            # Relative to itself is 0, but maybe we want Barycentric? 
            # Let's stick to Jovicentric for the game logic context.
            return np.zeros(3), np.zeros(3)
            
        target = self.moons.get(body_name.lower())
        if not target:
            raise ValueError(f"Unknown body: {body_name}")
            
        # Vector from Jupiter Center to Target
        # Note: jup365.bsp allows 'jupiter' center.
        # We calculate relative to Jupiter Center in ICRS frame.
        astrometric = (target - self.jupiter).at(t)
        pos = astrometric.position.km
        vel = astrometric.velocity.km_per_s
        
        return pos, vel

    def propagate(self, state_vector, time_iso, dt):
        """
        Propagate state vector (x,y,z,vx,vy,vz) in Jovicentric frame.
        state_vector: list or np.array of 6 floats
        time_iso: Start time string
        dt: Duration in seconds
        """
        from datetime import datetime, timezone
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        t_start = self.ts.from_datetime(dt_obj)
        y0 = np.array(state_vector)
        
        # Time span for integration
        t_span = (0, dt)
        
        # Pre-calculate body positions? 
        # For N-Body, bodies move. We need their position at t_start + t.
        # Calling Skyfield inside the integrator loop is slow (Python overhead).
        # Optimization: For short durations (hours), we could assume linear motion or quadratic?
        # NO, for "Hard SF" we want precision. We will call Skyfield. 
        # To optimize, we can cache the positions of moons at t_start and t_end and interpolate, 
        # but let's try direct call first.
        
        def equations_of_motion(t, y):
            # t is seconds from t_start
            current_time = t_start + (t / 86400.0) # Add days
            
            rx, ry, rz, vx, vy, vz = y
            r_ship = np.array([rx, ry, rz])
            
            # Acceleration due to Jupiter (Central Body)
            r_mag = np.linalg.norm(r_ship)
            a = -self.GM['jupiter'] * r_ship / (r_mag**3)
            
            # Perturbations from Moons and Sun
            # We need positions of Moons relative to Jupiter at current_time
            # Note: This is the bottleneck.
            
            # Optimization: We can query all at once? No.
            # Let's just do it.
            
            for name, moon_obj in self.moons.items():
                # Position of Moon relative to Jupiter
                r_moon = (moon_obj - self.jupiter).at(current_time).position.km
                
                # Vector from Moon to Ship
                r_moon_ship = r_ship - r_moon
                dist_moon_ship = np.linalg.norm(r_moon_ship)
                
                # Acceleration from Moon
                # Direct term: -GM * (r - r_moon) / |r - r_moon|^3
                # Indirect term (because frame is centered on Jupiter, which is also pulled by moons): 
                # -GM * r_moon / |r_moon|^3
                # (Encke's method logic or simply Non-Inertial frame correction)
                # Wait, Jovicentric frame is NON-INERTIAL because Jupiter is accelerating due to Sun/Moons.
                # However, for simulation simplicity, if we treat Jupiter as "fixed" origin, we miss the indirect term.
                # Correct N-Body in relative frame: a_rel = a_abs_ship - a_abs_jup
                
                gm = self.GM[name]
                a_direct = -gm * r_moon_ship / (dist_moon_ship**3)
                a_indirect = -gm * r_moon / (np.linalg.norm(r_moon)**3)
                
                a += (a_direct - a_indirect)
                
            # Sun perturbation
            r_sun = (self.sun - self.jupiter).at(current_time).position.km
            r_sun_ship = r_ship - r_sun
            dist_sun_ship = np.linalg.norm(r_sun_ship)
            gm_sun = self.GM['sun']
            
            a_direct = -gm_sun * r_sun_ship / (dist_sun_ship**3)
            a_indirect = -gm_sun * r_sun / (np.linalg.norm(r_sun)**3)
            a += (a_direct - a_indirect)

            return [vx, vy, vz, a[0], a[1], a[2]]

        sol = solve_ivp(
            equations_of_motion, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-6, 
            atol=1e-9
        )
        
        final_state = sol.y[:, -1]
        return final_state.tolist()

    def propagate_interpolated(self, state_vector, time_iso, duration, cache_step=300, t_eval=None):
        """
        Propagate state vector using interpolated body positions.
        
        Args:
            state_vector: Initial state [x, y, z, vx, vy, vz]
            time_iso: Start time in ISO format
            duration: Duration in seconds
            cache_step: Step size for caching body positions (seconds)
            t_eval: Optional list of time points (seconds from start) to evaluate at.
                    If provided, returns list of state vectors.
                    If None, returns final state vector.
        """
        from scipy.interpolate import CubicSpline
        from datetime import datetime, timezone, timedelta
        
        # Parse start time
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            
        # 1. Pre-calculate Body Positions (Splines)
        # We need to cover the full duration
        t_eval_steps = int(duration / cache_step) + 2 
        t_eval_seconds = np.linspace(0, duration + cache_step, t_eval_steps)
        
        t_list = [dt_obj + timedelta(seconds=s) for s in t_eval_seconds]
        ts_objects = self.ts.from_datetimes(t_list)
        
        body_splines = {}
        
        # Moons
        for name, body in self.moons.items():
            vectors = (body - self.jupiter).at(ts_objects).position.km
            body_splines[name] = CubicSpline(t_eval_seconds, vectors, axis=1)
            
        # Sun
        vectors = (self.sun - self.jupiter).at(ts_objects).position.km
        body_splines['sun'] = CubicSpline(t_eval_seconds, vectors, axis=1)
        
        # 2. Define Equations of Motion
        def equations(t, y):
            rx, ry, rz, vx, vy, vz = y
            r_ship = np.array([rx, ry, rz])
            
            # Jupiter (Central Body)
            r_mag = np.linalg.norm(r_ship)
            a = -self.GM['jupiter'] * r_ship / (r_mag**3)
            
            # Perturbations
            for name, spline in body_splines.items():
                r_body = spline(t)
                r_rel = r_ship - r_body
                dist = np.linalg.norm(r_rel)
                
                gm = self.GM[name] if name in self.GM else self.GM['sun']
                
                a_direct = -gm * r_rel / (dist**3)
                a_indirect = -gm * r_body / (np.linalg.norm(r_body)**3)
                a += (a_direct - a_indirect)
                
            return [vx, vy, vz, a[0], a[1], a[2]]

        # 3. Integrate
        y0 = np.array(state_vector)
        t_span = (0, duration)
        
        sol = solve_ivp(equations, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9)
        
        if t_eval is not None:
            return sol.y.T.tolist() # Return list of states
        else:
            return sol.y[:, -1].tolist() # Return final state

    def propagate_jit(self, state_vector, time_iso, duration, dt=10.0, cache_step=600, t_eval=None):
        """
        JIT-compiled propagation using Numba.
        
        Args:
            state_vector: Initial state [x, y, z, vx, vy, vz]
            time_iso: Start time in ISO format
            duration: Duration in seconds
            dt: Integration step size (seconds). Fixed step RK4.
            cache_step: Step size for caching body positions (seconds)
            t_eval: Optional list of time points (seconds from start) to evaluate at.
        """
        import numba_engine
        from scipy.interpolate import CubicSpline
        from datetime import datetime, timezone, timedelta
        
        # Parse start time
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            
        # 1. Pre-calculate Body Positions (Splines)
        t_eval_steps = int(duration / cache_step) + 2 
        t_eval_seconds = np.linspace(0, duration + cache_step, t_eval_steps)
        
        t_list = [dt_obj + timedelta(seconds=s) for s in t_eval_seconds]
        ts_objects = self.ts.from_datetimes(t_list)
        
        # Prepare arrays for Numba
        # Spline knots (shared)
        spline_x = np.ascontiguousarray(t_eval_seconds)
        
        # Sun Spline
        vectors_sun = (self.sun - self.jupiter).at(ts_objects).position.km
        spline_sun = CubicSpline(t_eval_seconds, vectors_sun, axis=1)
        spline_c_sun = np.ascontiguousarray(spline_sun.c) # (4, n_int, 3)
        
        # Moons Splines
        # We need to stack coefficients: (N_moons, 4, n_int, 3)
        moon_names = list(self.moons.keys())
        n_moons = len(moon_names)
        n_int = spline_c_sun.shape[1]
        
        spline_c_moons = np.empty((n_moons, 4, n_int, 3), dtype=np.float64)
        gm_moons = np.empty(n_moons, dtype=np.float64)
        
        for i, name in enumerate(moon_names):
            vectors = (self.moons[name] - self.jupiter).at(ts_objects).position.km
            spline = CubicSpline(t_eval_seconds, vectors, axis=1)
            spline_c_moons[i] = spline.c
            gm_moons[i] = self.GM[name]
            
        spline_c_moons = np.ascontiguousarray(spline_c_moons)
        
        # 2. Call Numba Propagator
        state_0 = np.array(state_vector, dtype=np.float64)
        
        t_eval_arr = None
        if t_eval is not None:
            t_eval_arr = np.ascontiguousarray(t_eval, dtype=np.float64)
            
        res = numba_engine.propagate_numba_loop(
            state_0, 0.0, duration, dt,
            self.GM['jupiter'], self.GM['sun'], gm_moons,
            spline_x, spline_c_sun, spline_c_moons,
            t_eval=t_eval_arr
        )
        
        if t_eval is not None:
            return res.tolist()
        else:
            return res[0].tolist()
