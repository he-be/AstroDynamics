import os
import numpy as np
from skyfield.api import Loader, Topos


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
        
        # Heyoka Integrator Cache
        self.ta = None
        self.ta_controlled = None

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

    def propagate(self, state_vector, time_iso, duration, t_eval=None, order=20):
        """
        Propagate state vector using Heyoka (Taylor Series Integrator).
        This is the primary propagation method.
        
        Args:
            state_vector: Initial state [x, y, z, vx, vy, vz] (Jovicentric frame)
            time_iso: Start time in ISO format
            duration: Duration in seconds
            t_eval: Optional list of time points (seconds from start) to evaluate at.
            order: Taylor series order (default 20).
            
        Returns:
            Final state vector (list) or list of state vectors if t_eval is provided.
        """
        try:
            import heyoka as hy
        except ImportError:
            raise ImportError("Heyoka not found. Please install 'heyoka.py' in your environment.")
            
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
        
        # 3. Setup Integrator / Reuse Cache
        # Ensure floats
        y0 = [float(x) for x in y0]
        gms = [float(x) for x in gms]
        
        if self.ta is None:
            # Increase recursion limit to handle complex N-body expression printing if errors occur
            import sys
            sys.setrecursionlimit(10000)
            
            sys = hy.model.nbody(len(gms), masses=gms)
            self.ta = hy.taylor_adaptive(sys, y0)
        else:
            # Reuse existing integrator (AVOID RECOMPILATION)
            self.ta.time = 0.0
            self.ta.state[:] = y0
        
        # 4. Propagate
        if t_eval is not None:
            # Propagate to specific points
            grid = np.array(t_eval, dtype=np.float64)
            # Ensure strictly increasing and starting >= 0
            if grid[0] < 0:
                 raise ValueError("t_eval must be >= 0")
                 
            # propagate_grid returns (times, data)
            # data shape: (N_points, N_vars)
            res = self.ta.propagate_grid(grid)
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
            self.ta.propagate_until(float(duration))
            
            # Extract final state from integrator
            # ta.state is current state vector
            final_y = self.ta.state
            
            # Extract Jup (0-6) and Ship (last 6)
            p_jup = final_y[0:3]
            v_jup = final_y[3:6]
            
            p_ship = final_y[-6:-3]
            v_ship = final_y[-3:]
            
            # Relative
            p_rel = np.array(p_ship) - np.array(p_jup)
            v_rel = np.array(v_ship) - np.array(v_jup)
            
            return np.concatenate([p_rel, v_rel]).tolist()

    def propagate_controlled(self, state_vector, time_iso, duration, thrust_vector, mass, isp):
        """
        Propagate with Finite Burn (Constant Thrust).
        
        Args:
            state_vector: [x,y,z,vx,vy,vz] (Jovicentric).
            time_iso: Start Time.
            duration: Burn duration (s).
            thrust_vector: [Fx, Fy, Fz] in Newtons (Inertial Frame).
            mass: Initial Mass (kg).
            isp: Specific Impulse (s).
            
        Returns:
            (final_state_vector, final_mass)
        """
        try:
            import heyoka as hy
        except ImportError:
            raise ImportError("Heyoka not found.")
            
        from datetime import datetime, timezone
        
        # 1. Parse Time & State (Same as propagate)
        if time_iso.endswith('Z'): time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        t_start = self.ts.from_datetime(dt_obj)
        
        y0 = []
        gms = []
        
        # Bodies setup (Jupiter, Moons, Sun, Ship)
        # Note: Code duplication with propagate(). Could refactor, but for now inline for clarity.
        
        # Jup Bary
        jb = self.planets['jupiter barycenter'].at(t_start)
        p_jb, v_jb = jb.position.km, jb.velocity.km_per_s
        # Jup Center
        jc = self.jup_moons['jupiter'].at(t_start)
        p_jc, v_jc = jc.position.km, jc.velocity.km_per_s
        
        p_jup_ssb = p_jb + p_jc
        v_jup_ssb = v_jb + v_jc
        
        y0.extend(p_jup_ssb); y0.extend(v_jup_ssb); gms.append(self.GM['jupiter'])
        
        for name in ['io', 'europa', 'ganymede', 'callisto']:
            b = self.jup_moons[name].at(t_start)
            y0.extend(p_jb + b.position.km)
            y0.extend(v_jb + b.velocity.km_per_s)
            gms.append(self.GM[name])
            
        sun = self.planets['sun'].at(t_start)
        y0.extend(sun.position.km); y0.extend(sun.velocity.km_per_s); gms.append(self.GM['sun'])
        
        # Ship
        p_s_j = np.array(state_vector[0:3])
        v_s_j = np.array(state_vector[3:6])
        y0.extend(p_jup_ssb + p_s_j)
        y0.extend(v_jup_ssb + v_s_j)
        gms.append(0.0)
        
        # Add Initial Mass to State
        y0.append(float(mass))
        y0 = [float(x) for x in y0]
        gms = [float(x) for x in gms]
        
        # 2. Setup/Reuse Controlled Integrator
        if self.ta_controlled is None:
            import sys; sys.setrecursionlimit(10000)
            
            # Base N-Body System
            sys_eq = hy.model.nbody(len(gms), masses=gms)
            
            # Define Parameters and Variables
            # Use heyoka.par[index] for runtime parameters
            tx = hy.par[0]
            ty = hy.par[1]
            tz = hy.par[2]
            m_dot = hy.par[3]
            m_ship = hy.make_vars("m_ship") # Mass variable (State)
            
            # Modify Ship Acceleration (Index 6)
            # Indices in sys_eq: 6*6 + 3,4,5 = 39, 40, 41
            # sys_eq[39] is (vx, ax). We want new_ax = ax + tx/(m*1000)
            # Divide by 1000 to convert m/s^2 to km/s^2
            
            for i, param_force in zip([39, 40, 41], [tx, ty, tz]):
                var, expr = sys_eq[i]
                new_expr = expr + (param_force / (m_ship * 1000.0))
                sys_eq[i] = (var, new_expr)
                
            # Add Mass Equation: dm/dt = -m_dot
            sys_eq.append((m_ship, -m_dot))
            
            self.ta_controlled = hy.taylor_adaptive(sys_eq, y0)
            
        # 3. Update Parameters & State
        # Calculate Flow Rate
        thrust_mag = np.linalg.norm(thrust_vector)
        g0 = 9.80665
        flow_rate = thrust_mag / (isp * g0)

        # Update Params: [tx, ty, tz, m_dot]
        self.ta_controlled.pars[:] = [thrust_vector[0], thrust_vector[1], thrust_vector[2], flow_rate]
        
        # Update State
        self.ta_controlled.time = 0.0
        self.ta_controlled.state[:] = y0
        
        # 4. Propagate
        self.ta_controlled.propagate_until(float(duration))
        
        # 5. Extract Result
        final = self.ta_controlled.state
        
        # Ship: Indices 36-42 (pos, vel). Mass is last (42).
        # Jup: 0-6.
        # Relative State
        p_jup = final[0:3]
        v_jup = final[3:6]
        
        p_ship = final[-7:-4] # x,y,z of ship (before mass)
        v_ship = final[-4:-1] # vx,vy,vz of ship
        final_mass = final[-1] # last element is mass
        
        p_rel = np.array(p_ship) - np.array(p_jup)
        v_rel = np.array(v_ship) - np.array(v_jup)
        
        final_state = np.concatenate([p_rel, v_rel]).tolist()
        
        return final_state, float(final_mass)
