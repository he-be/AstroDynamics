import os
import numpy as np
from skyfield.api import Loader, Topos


class PhysicsEngine:
    def __init__(self, data_dir='./data', include_saturn=True):
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
        self.saturn_bary = self.planets['saturn barycenter']
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
            'saturn': 3.7931187e7,
            'io': 5959.91,
            'europa': 3202.73,
            'ganymede': 9887.83,
            'callisto': 7179.28
        }
        
        self.include_saturn = include_saturn
        
        # Heyoka Integrator Cache
        self.ta = None
        self.ta = None
        self.ta_controlled = None
        self.ta_controlled_var = None
        self.ta_controlled_lts = None
        self.ta_controlled_lts_var = None

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
            if body_name.lower() == 'sun':
                target = self.planets['sun']
            elif body_name.lower() == 'saturn':
                target = self.planets['saturn barycenter']
            else:
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
            body = self.jup_moons[name].at(t_start)
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
        
        if self.include_saturn:
            # 2d. Saturn (SSB)
            sat = self.planets['saturn barycenter']
            s_sat = sat.at(t_start)
            y0.extend(s_sat.position.km)
            y0.extend(s_sat.velocity.km_per_s)
            gms.append(self.GM['saturn'])
        
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
            
            # Ship is the last body in the N-body system
            # Its state is at index (num_bodies - 1) * 6
            ship_body_idx = len(gms) - 1
            ship_state_idx = ship_body_idx * 6
            
            p_ship = final_y[ship_state_idx : ship_state_idx+3]
            v_ship = final_y[ship_state_idx+3 : ship_state_idx+6]
            
            # Relative
            p_rel = np.array(p_ship) - np.array(p_jup)
            v_rel = np.array(v_ship) - np.array(v_jup)
            
            return np.concatenate([p_rel, v_rel]).tolist()

    def propagate_controlled(self, state_vector, time_iso, duration, control_params, mass, isp, with_variational_equations=False, steering_mode='constant', thrust_magnitude=None):
        """
        Propagate with Finite Burn.
        """
        try:
            if steering_mode == 'linear_tangent' and control_params is not None:
                # print(f"DEBUG: Engine Params: {control_params[0]:.4f} ...")
                pass 
            import heyoka as hy
        except ImportError:
            raise ImportError("Heyoka not found.")
            
        from datetime import datetime, timezone
        
        # 1. Parse Time & State
        if time_iso.endswith('Z'): time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        t_start = self.ts.from_datetime(dt_obj)
        
        y0 = []
        gms = []
        
        # Bodies Setup
        jb = self.planets['jupiter barycenter'].at(t_start)
        p_jb, v_jb = jb.position.km, jb.velocity.km_per_s
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
        
        if self.include_saturn:
            sat = self.planets['saturn barycenter'].at(t_start)
            y0.extend(sat.position.km); y0.extend(sat.velocity.km_per_s); gms.append(self.GM['saturn'])
        
        # Ship
        p_s_j = np.array(state_vector[0:3])
        v_s_j = np.array(state_vector[3:6])
        y0.extend(p_jup_ssb + p_s_j)
        y0.extend(v_jup_ssb + v_s_j)
        gms.append(0.0)
        
        y0.append(float(mass))
        y0 = [float(x) for x in y0]
        gms = [float(x) for x in gms]
        
        integrator = None
        
        # 2. Setup Integrator
        if steering_mode == 'linear_tangent':
            needs_init = False
            if with_variational_equations:
                if self.ta_controlled_lts_var is None: needs_init = True
            else:
                if self.ta_controlled_lts is None: needs_init = True
            
            if needs_init:
                import sys; sys.setrecursionlimit(10000)
                sys_eq = hy.model.nbody(len(gms), masses=gms)
                
                # Parameters: ax, ay, az, bx, by, bz, m_dot, F_mag
                ax, ay, az = hy.par[0], hy.par[1], hy.par[2]
                bx, by, bz = hy.par[3], hy.par[4], hy.par[5]
                m_dot_par = hy.par[6]
                F_mag_par = hy.par[7]
                
                m_ship = hy.make_vars("m_ship")
                
                # Thrust Vector Construction
                tv_x = ax + bx * hy.time
                tv_y = ay + by * hy.time
                tv_z = az + bz * hy.time
                tv_norm = hy.sqrt(tv_x**2 + tv_y**2 + tv_z**2)
                
                dir_x = tv_x / tv_norm
                dir_y = tv_y / tv_norm
                dir_z = tv_z / tv_norm
                
                force_acc = F_mag_par / (m_ship * 1000.0)
                
                ship_idx = len(gms) - 1
                base_idx = ship_idx * 6
                
                # Acceleration
                for i, acc_expr in zip([base_idx+3, base_idx+4, base_idx+5], [force_acc*dir_x, force_acc*dir_y, force_acc*dir_z]):
                    var, expr = sys_eq[i]
                    sys_eq[i] = (var, expr + acc_expr)
                
                sys_eq.append((m_ship, -m_dot_par))
                
                if with_variational_equations:
                    # Differentiate w.r.t first 6 parameters (ax..bz)
                    var_sys = hy.var_ode_sys(sys_eq, [hy.par[0], hy.par[1], hy.par[2], hy.par[3], hy.par[4], hy.par[5]])
                    self.ta_controlled_lts_var = hy.taylor_adaptive(var_sys, y0)
                else:
                    self.ta_controlled_lts = hy.taylor_adaptive(sys_eq, y0)
            
            integrator = self.ta_controlled_lts_var if with_variational_equations else self.ta_controlled_lts
            
            # Setup Params
            ax, ay, az = control_params[0:3]
            bx, by, bz = control_params[3:6]
            
            if thrust_magnitude is None: raise ValueError("thrust_magnitude required for LTS")
            g0 = 9.80665
            flow_rate = thrust_magnitude / (isp * g0)
            
            integrator.pars[:] = [ax, ay, az, bx, by, bz, flow_rate, thrust_magnitude]
            
            integrator.time = 0.0
            
            if with_variational_equations:
                # Initialize full state for var eqs
                n_states = len(y0)
                full_state = np.zeros(len(integrator.state))
                full_state[0:n_states] = y0
                integrator.state[:] = full_state
            else:
                integrator.state[:] = y0
            
        else:
            # Constant Steering
            if self.ta_controlled is None and not with_variational_equations:
                 import sys; sys.setrecursionlimit(10000)
                 sys_eq = hy.model.nbody(len(gms), masses=gms)
                 tx, ty, tz, m_dot = hy.par[0], hy.par[1], hy.par[2], hy.par[3]
                 m_ship = hy.make_vars("m_ship")
                 ship_idx = len(gms) - 1
                 base_idx = ship_idx * 6
                 for i, param_force in zip([base_idx+3, base_idx+4, base_idx+5], [tx, ty, tz]):
                     var, expr = sys_eq[i]
                     new_expr = expr + (param_force / (m_ship * 1000.0))
                     sys_eq[i] = (var, new_expr)
                 sys_eq.append((m_ship, -m_dot))
                 self.ta_controlled = hy.taylor_adaptive(sys_eq, y0)
            
            if with_variational_equations:
                 if self.ta_controlled_var is None:
                     import sys; sys.setrecursionlimit(10000)
                     sys_eq = hy.model.nbody(len(gms), masses=gms)
                     tx, ty, tz, m_dot = hy.par[0], hy.par[1], hy.par[2], hy.par[3]
                     m_ship = hy.make_vars("m_ship")
                     ship_idx = len(gms) - 1
                     base_idx = ship_idx * 6
                     for i, param_force in zip([base_idx+3, base_idx+4, base_idx+5], [tx, ty, tz]):
                         var, expr = sys_eq[i]
                         new_expr = expr + (param_force / (m_ship * 1000.0))
                         sys_eq[i] = (var, new_expr)
                     sys_eq.append((m_ship, -m_dot))
                     
                     state_vars = [v for v, _ in sys_eq]
                     params_vars = [hy.par[0], hy.par[1], hy.par[2]]
                     vos = hy.var_ode_sys(sys_eq, args=state_vars + params_vars)
                     sys_eq_aug = vos.sys
                     
                     n_states = len(y0)
                     n_params = 3
                     stm_init_state = np.eye(n_states).flatten()
                     stm_init_params = np.zeros(n_states * n_params)
                     y0_var = np.concatenate([y0, stm_init_state, stm_init_params])
                     self.ta_controlled_var = hy.taylor_adaptive(sys_eq_aug, y0_var)
                     
                 integrator = self.ta_controlled_var
            else:
                 integrator = self.ta_controlled
            
            thrust_vec = control_params
            thrust_mag = np.linalg.norm(thrust_vec)
            g0 = 9.80665
            flow_rate = thrust_mag / (isp * g0)
            
            integrator.pars[:] = [thrust_vec[0], thrust_vec[1], thrust_vec[2], flow_rate]
            
            if with_variational_equations:
                 n_states = len(y0)
                 stm_init_state = np.eye(n_states).flatten()
                 stm_init_params = np.zeros(n_states * 3)
                 y0_var = np.concatenate([y0, stm_init_state, stm_init_params])
                 integrator.time = 0.0
                 integrator.state[:] = y0_var
            else:
                 integrator.time = 0.0
                 integrator.state[:] = y0

        # Propagate
        integrator.propagate_until(float(duration))
        final = integrator.state
        
        # Ship Extraction
        p_jup = final[0:3]
        v_jup = final[3:6]
        
        # Ship Index Calculation
        ship_body_idx = len(gms) - 1
        ship_state_idx = ship_body_idx * 6
        
        p_ship = final[ship_state_idx : ship_state_idx+3]
        v_ship = final[ship_state_idx+3 : ship_state_idx+6]
        final_mass = final[ship_state_idx+6]
        
        p_rel = np.array(p_ship) - np.array(p_jup)
        v_rel = np.array(v_ship) - np.array(v_jup)
        
        final_state = np.concatenate([p_rel, v_rel]).tolist()
        
        if with_variational_equations:
             if steering_mode == 'linear_tangent':
                 # Extract Jacobian (3x6) for position
                 n_sys = len(y0)
                 jac = np.zeros((3, 6))
                 
                 for j in range(6): # 6 params: ax, ay, az, bx, by, bz
                      # Start index of variations for param j
                      # The variations segments are appended in order of params (0 to 5)
                      # Each segment has length n_sys
                      var_start = n_sys + j * n_sys
                      
                      # Derivatives of Ship Position [x, y, z] at ship_state_idx
                      jac[0, j] = final[var_start + ship_state_idx]
                      jac[1, j] = final[var_start + ship_state_idx + 1]
                      jac[2, j] = final[var_start + ship_state_idx + 2]
                 
                 return final_state, float(final_mass), jac
             else:
                 return final_state, float(final_mass), final
        else:
             return final_state, float(final_mass)


