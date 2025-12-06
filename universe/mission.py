import numpy as np
from datetime import datetime, timedelta
from planning import solve_lambert

class MissionPlanner:
    def __init__(self, engine):
        self.engine = engine
        self.mu = engine.GM['jupiter'] # Default to Jupiter context

    def calculate_transfer(self, r_start, r_target, dt, mu=None):
        """
        Solves Lambert problem to find required Departure and Arrival velocities.
        
        Args:
            r_start: Position vector (km)
            r_target: Position vector (km)
            dt: Time of flight (seconds)
            mu: Gravitational parameter (km^3/s^2). Defaults to Jupiter.
            
        Returns:
            v_dep_vec (km/s), v_arr_vec (km/s)
        """
        if mu is None: mu = self.mu
        
        # solve_lambert returns v_dep_vec, v_arr_vec
        v_dep, v_arr_transfer = solve_lambert(r_start, r_target, dt, mu)
        return v_dep, v_arr_transfer

    def verify_fuel(self, total_dv_km_s, mass, isp):
        """
        Verify if fuel is sufficient for Total Delta V.
        Args:
            total_dv_km_s: Delta V in km/s.
            mass: Wet mass (kg).
            isp: Specific Impulse (s).
        Returns:
            (is_ok, fuel_consumed_kg)
        """
        g0 = 9.80665
        # Convert km/s to m/s
        dv_meters = total_dv_km_s * 1000.0
        mass_final = mass * np.exp(-dv_meters / (isp * g0))
        return mass_final > 0, mass - mass_final

class FlightController:
    def __init__(self, engine):
        self.engine = engine
        self.state = None # [rx,ry,rz, vx,vy,vz] in Jovicentric frame
        self.mass = None
        self.time_iso = None
        self._trajectory_log = [] # List of {'time': iso_str, 'state': [6], 'mass': float}
        self.maneuver_log = [] # List of {t, dv, duration, label, type}

    @property
    def trajectory_log(self):
        """Backward compatibility for plotting: returns list of [x,y,z]"""
        return [entry['state'][:3] for entry in self._trajectory_log]

    def set_initial_state(self, state, mass, time_iso):
        self.state = state
        self.mass = mass
        self.time_iso = time_iso
        self._trajectory_log.append({
            'time': time_iso,
            'state': state,
            'mass': mass
        })

    def execute_burn(self, plan_dv_vec_km_s, thrust_force, isp, label="Burn"):
        """
        Executes a finite burn matching the requested Delta V vector (km/s).
        """
        dv_mag_km = np.linalg.norm(plan_dv_vec_km_s)
        if dv_mag_km < 1e-9:
            # print(f"[{label}] Delta V negligible. Skipping.")
            return

        # Calculate Duration (Tsiolkovsky)
        # t = m_in * ve * (1 - exp(-dv/ve)) / F
        g0 = 9.80665
        ve = isp * g0 # m/s
        dv_meters = dv_mag_km * 1000.0
        
        m_in = self.mass
        duration = (m_in * ve * (1 - np.exp(-dv_meters/ve))) / thrust_force
        
        print(f"[{label}] Executing DV={dv_meters:.2f} m/s. Duration={duration:.2f} s. Time={self.time_iso}")
        
        self.maneuver_log.append({
            'time_iso': self.time_iso,
            'delta_v_m_s': dv_meters,
            'delta_v_vec_km_s': plan_dv_vec_km_s.tolist(),
            'duration_s': duration,
            'label': label,
            'type': 'finite',
            'mass_before': m_in
        })
        
        # Thrust Vector Direction
        thrust_dir = plan_dv_vec_km_s / dv_mag_km
        thrust_vec = thrust_dir * thrust_force
        
        # Execute via Engine
        state_new, mass_new = self.engine.propagate_controlled(
            self.state, self.time_iso, duration,
            thrust_vector=thrust_vec.tolist(),
            mass=self.mass,
            isp=isp
        )
        
        # Update Internal State
        self.state = state_new
        self.mass = mass_new
        self.update_time(duration)
        
        # Log final state of burn
        self._trajectory_log.append({
            'time': self.time_iso,
            'state': self.state,
            'mass': self.mass
        })
        
    def coast(self, duration, step_points=100):
        # print(f"[Coast] Drifting for {duration:.1f} s...")
        if duration <= 1e-3: return

        t_eval = np.linspace(0, duration, step_points)
        
        # PhysicsEngine now returns full states [x,y,z,vx,vy,vz]
        states = self.engine.propagate(
            self.state, self.time_iso, duration, t_eval=t_eval
        )
        
        start_dt = datetime.fromisoformat(self.time_iso.replace('Z', '+00:00'))
        
        for i, s in enumerate(states):
            # Calculate time for this step
            t_offset = t_eval[i]
            dt = start_dt + timedelta(seconds=t_offset)
            t_iso = dt.isoformat().replace('+00:00', 'Z')
            
            self._trajectory_log.append({
                'time': t_iso,
                'state': s,
                'mass': self.mass
            })
            
        self.state = states[-1]
        self.update_time(duration)

    def update_time(self, seconds_added):
        dt_obj = datetime.fromisoformat(self.time_iso.replace('Z', '+00:00'))
        dt_obj += timedelta(seconds=seconds_added)
        self.time_iso = dt_obj.isoformat().replace('+00:00', 'Z')
        
    def get_position(self):
        return np.array(self.state[:3])
    
    def get_velocity(self):
        return np.array(self.state[3:6])
