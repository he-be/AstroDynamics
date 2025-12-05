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

    def propagate_heyoka(self, state_vector, time_iso, duration, t_eval=None, order=20):
        """
        Fast High-Precision propagation using Heyoka (Taylor Series Integrator).
        Requires 'heyoka.py' installed (e.g. in Conda environment).
        
        Args:
            state_vector: Initial state [x, y, z, vx, vy, vz] (Jovicentric frame)
            time_iso: Start time in ISO format
            duration: Duration in seconds
            t_eval: Optional list of time points (seconds from start) to evaluate at.
            order: Taylor series order (default 20). Lower (e.g. 10) for faster compilation.
            
        Returns:
            Final state vector (list) or list of state vectors if t_eval is provided.
        """
        try:
            import heyoka as hy
        except ImportError:
            raise ImportError("Heyoka not found. Please install 'heyoka.py' in your environment (see docs/environment_setup.md).")
            
        from datetime import datetime, timezone
        
        # 1. Parse Time
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        t_start = self.ts.from_datetime(dt_obj)
        
        # 2. Setup N-Body System (Relative to SSB)
        # We need to construct the full system state y0
        # Bodies: Jupiter (0), Moons (1-4), Sun (5), Ship (6)
        
        y0 = []
        gms = []
        
        # 2a. Jupiter Barycenter (SSB)
        jup_bary = self.planets['jupiter barycenter']
        s_jb = jup_bary.at(t_start)
        p_jb = s_jb.position.km
        v_jb = s_jb.velocity.km_per_s
        
        # Jupiter Center (relative to Jup Bary)
        jup_center = self.jup_moons['jupiter']
        s_jc = jup_center.at(t_start)
        p_jc = s_jc.position.km
        v_jc = s_jc.velocity.km_per_s
        
        # Jupiter Absolute (SSB)
        p_jup_ssb = p_jb + p_jc
        v_jup_ssb = v_jb + v_jc
        
        y0.extend(p_jup_ssb)
        y0.extend(v_jup_ssb)
        gms.append(self.GM['jupiter'])
        
        # 2b. Moons (SSB)
        # Order: Io, Europa, Ganymede, Callisto
        moon_names = ['io', 'europa', 'ganymede', 'callisto']
        for name in moon_names:
            body = self.jup_moons[name]
            s_b = body.at(t_start) # Relative to Jup Bary? No, Relative to Kernel center?
            # Skyfield .at() returns position relative to SSB if chained correctly?
            # Creating 'body' from 'self.jup_moons' (loaded from jup365.bsp).
            # Usually .at() does the lookup. But to be safe and match demo:
            # We chain: Jup Bary -> Moon (from kernel).
            # Wait, `body.at(t)` computes position relative to Solar System Barycenter IF `body` is linked to it.
            # But `self.jup_moons` is just a file.
            # `self.jup_moons['io']` is a target.
            # `(self.jup_moons['io'] - self.planets['jupiter barycenter']).at(t)` ? No.
            # In demo we did: `p_b_ssb = p_jb_ssb + s_b_jb.position.km` provided `s_b_jb` is relative to Jup Bary?
            # Let's use the robust `(body - SSB).at(t)` if possible, but we don't have SSB object handy?
            # Actually `self.planets` (de440s) has SSB? Usually segment 0.
            # Let's stick to the manual chaining from the demo which we verified.
            
            # Moon relative to Jup Barycenter (default for jup365)
            s_b_jb = body.at(t_start) 
            p_b_ssb = p_jb + s_b_jb.position.km
            v_b_ssb = v_jb + s_b_jb.velocity.km_per_s
            
            y0.extend(p_b_ssb)
            y0.extend(v_b_ssb)
            gms.append(self.GM[name])
            
        # 2c. Sun (SSB)
        sun = self.planets['sun']
        s_sun = sun.at(t_start) # Rel to SSB (de440s)
        y0.extend(s_sun.position.km)
        y0.extend(s_sun.velocity.km_per_s)
        gms.append(self.GM['sun'])
        
        # 2d. Ship (SSB)
        # Input `state_vector` is Jovicentric (Relative to Jupiter Center).
        # Ship_SSB = Jup_SSB + Ship_ICRS
        p_ship_jup = np.array(state_vector[0:3])
        v_ship_jup = np.array(state_vector[3:6])
        
        p_ship_ssb = p_jup_ssb + p_ship_jup
        v_ship_ssb = v_jup_ssb + v_ship_jup
        
        y0.extend(p_ship_ssb)
        y0.extend(v_ship_ssb)
        gms.append(0.0) # Massless ship
        
        # 3. Setup Integrator
        # Ensure floats
        y0 = [float(x) for x in y0]
        gms = [float(x) for x in gms]
        
        sys = hy.model.nbody(len(gms), masses=gms)
        ta = hy.taylor_adaptive(sys, y0, order=order) # Use kwargs compatible order
        
        # 4. Propagate
        if t_eval is not None:
            # Propagate to specific points
            grid = np.array(t_eval, dtype=np.float64)
            # Ensure strictly increasing and starting >= 0
            if grid[0] < 0:
                 raise ValueError("t_eval must be >= 0")
                 
            # propagate_grid returns (times, data)
            # data shape: (N_points, N_vars)
            res = ta.propagate_grid(grid)
            data = res[5] # Index 5 based on debug findings
            
            # Convert back to Jovicentric
            # Data structure: [Jup(6), Moons(4*6), Sun(6), Ship(6)]
            # Ship is last 6 columns. Jup is first 6.
            # Ship_Jup = Ship_SSB - Jup_SSB
            
            # Extract Jup and Ship states from all time steps
            jup_cols = slice(0, 6)
            ship_cols = slice(-6, None)
            
            jup_states = data[:, jup_cols]
            ship_states = data[:, ship_cols]
            
            rel_states = ship_states - jup_states
            return rel_states.tolist()
            
        else:
            # Propagate to final time
            ta.propagate_until(float(duration))
            
            # Extract final state from integrator
            # ta.state is current state vector
            final_y = ta.state
            
            # Extract Jup (0-6) and Ship (last 6)
            p_jup = final_y[0:3]
            v_jup = final_y[3:6]
            
            p_ship = final_y[-6:-3]
            v_ship = final_y[-3:]
            
            # Relative
            p_rel = np.array(p_ship) - np.array(p_jup)
            v_rel = np.array(v_ship) - np.array(v_jup)
            
            return np.concatenate([p_rel, v_rel]).tolist()
