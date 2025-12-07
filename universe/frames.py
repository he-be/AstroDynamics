import numpy as np

class FrameManager:
    def __init__(self, engine):
        self.engine = engine
        
    def transform_to_rotating(self, state_inertial, time_iso, center_body, secondary_body):
        """
        Transform state from Jovicentric Inertial (ICRS) to Synodic Rotating Frame.
        
        Frame Definition:
        - Center: center_body (e.g. Jupiter)
        - X-axis: Points from center_body to secondary_body (e.g. Ganymede)
        - Z-axis: Parallel to orbital angular momentum of secondary
        - Y-axis: Completes right-handed system (roughly along velocity)
        
        Args:
            state_inertial: [x,y,z,vx,vy,vz] relative to center_body (Inertial).
            time_iso: Time string.
            center_body: Name of primary (e.g. 'jupiter')
            secondary_body: Name of secondary (e.g. 'ganymede')
            
        Returns:
            state_rot: [x,y,z,vx,vy,vz] in Rotating Frame.
        """
        # 1. Get States
        # Note: engine.get_body_state returns Jovicentric state?
        # Yes, get_body_state returns relative to Jupiter Center?
        # Let's verify Engine API.
        # "get_body_state ... returns state relative to SSB? or Jupiter?"
        # engine.get_body_state docstring says: "Returns ... relative to Jupiter Center (if body_name != jupiter)?"
        # Wait, if center_body is 'jupiter', engine.get_body_state('ganymede') gives Jovicentric.
        # If center_body is NOT jupiter, we might need manual subtraction.
        # Let's assume standard usage: center='jupiter', secondary='ganymede'.
        
        # Primary State (usually [0,0,0] if center='jupiter')
        if center_body == 'jupiter':
            p_pri = np.zeros(3)
            v_pri = np.zeros(3)
        else:
            p_pri_full, v_pri_full = self.engine.get_body_state(center_body, time_iso)
            p_pri = np.array(p_pri_full)
            v_pri = np.array(v_pri_full)
            
        # Secondary State
        p_sec_full, v_sec_full = self.engine.get_body_state(secondary_body, time_iso)
        p_sec = np.array(p_sec_full)
        v_sec = np.array(v_sec_full)
        
        # Relative Secondary State (r_vec, v_vec)
        r_sec = p_sec - p_pri
        v_sec = v_sec - v_pri
        
        # 2. Define Axes
        r_mag = np.linalg.norm(r_sec)
        x_hat = r_sec / r_mag
        
        h_vec = np.cross(r_sec, v_sec)
        h_mag = np.linalg.norm(h_vec)
        z_hat = h_vec / h_mag
        
        y_hat = np.cross(z_hat, x_hat)
        
        # Rotation Matrix (Inertial -> Rotating)
        # Row 0: x_hat
        # Row 1: y_hat
        # Row 2: z_hat
        R = np.array([x_hat, y_hat, z_hat])
        
        # 3. Transform Position
        # Input State (relative to Center Body already?)
        # Argument says "state_inertial ... relative to center_body".
        # So input is [rx, ...].
        p_in = np.array(state_inertial[0:3])
        v_in = np.array(state_inertial[3:6])
        
        p_rot = R @ p_in
        
        # 4. Transform Velocity
        # v_rot = R @ v_in - omega x p_rot
        # Calculate angular velocity vector omega
        # omega = (r x v) / r^2 ? No.
        # omega vector direction is z_hat.
        # Magnitude w = h / r^2 (for circular).
        # General: omega = (r x v) / r^2 ??
        # Instantaneous angular velocity of the frame (defined by r).
        # The frame X-axis tracks r.
        # Rotation rate of r vector is |v_perp| / r.
        # v_perp = v - dot(v, x_hat)*x_hat.
        # w = |v_perp| / r.
        # Direction is z_hat (mostly).
        # Actually simplest: omega_vec_inertial = cross(r, v) / r^2.
        # Wait, cross(r, v) is h.
        # So omega = h / r^2.
        omega_vec_inertial = h_vec / (r_mag**2)
        
        # Transform omega to rotating frame?
        # omega_rot = R @ omega_vec_inertial
        # Should be [0, 0, w].
        omega_rot = R @ omega_vec_inertial
        
        # Coriolis term
        # v_rotating = v_in_rot - cross(omega_rot, p_rot)
        # where v_in_rot = R @ v_in
        v_in_rot = R @ v_in
        transport_vel = np.cross(omega_rot, p_rot)
        
        v_rot = v_in_rot - transport_vel
        
        return np.concatenate([p_rot, v_rot]).tolist()

    def get_lagrange_point(self, point_name, time_iso, center_body, secondary_body):
        """
        Calculate inertial state of Lagrange points (L1, L2 approximated).
        
        Args:
            point_name: 'L1' or 'L2'
            time_iso: Time ISO string.
            center_body: Primary name.
            secondary_body: Secondary name.
            
        Returns:
            [x,y,z,vx,vy,vz] in Inertial Jovicentric frame.
        """
        # 1. Get Mass Ratio
        GM_pri = self.engine.GM.get(center_body)
        GM_sec = self.engine.GM.get(secondary_body)
        
        if GM_pri is None or GM_sec is None:
            raise ValueError("Unknown body GM")
            
        # Alpha (Hill sphere scaler)
        alpha = (GM_sec / (3 * GM_pri))**(1/3.0)
        
        # 2. Get States
        # If center is jupiter, p_pri is 0? Use get_body_state logic
        if center_body == 'jupiter':
            p_pri = np.zeros(3)
            v_pri = np.zeros(3)
        else:
            p_pri_full, v_pri_full = self.engine.get_body_state(center_body, time_iso)
            p_pri = np.array(p_pri_full)
            v_pri = np.array(v_pri_full)
            
        p_sec_full, v_sec_full = self.engine.get_body_state(secondary_body, time_iso)
        p_sec = np.array(p_sec_full)
        v_sec = np.array(v_sec_full)
        
        # Vectors relative to Primary
        r_vec = p_sec - p_pri
        v_vec = v_sec - v_pri
        r_mag = np.linalg.norm(r_vec)
        
        # 3. Calculate State
        # L1/L2/L3 logic (Collinear)
        if point_name in ['L1', 'L2', 'L3']:
            if point_name == 'L1':
                scale = 1.0 - alpha
            elif point_name == 'L2':
                scale = 1.0 + alpha
            elif point_name == 'L3':
                scale = -(1.0 - (7.0/12.0) * (GM_sec/GM_pri)) # Approx -1
            
            p_L = p_pri + r_vec * scale
            v_L = v_pri + v_vec * scale
            
        # L4/L5 logic (Triangular)
        elif point_name in ['L4', 'L5']:
            # Rotation around angular momentum vector (z_hat)
            h_vec = np.cross(r_vec, v_vec)
            z_hat = h_vec / np.linalg.norm(h_vec)
            
            # Angle: L4 is +60 deg, L5 is -60 deg
            angle = np.pi/3.0 if point_name == 'L4' else -np.pi/3.0
            c = np.cos(angle)
            s = np.sin(angle)
            
            # Rodrigues formula (simplified since r, v are perpendicular to z_hat)
            # v_rot = v*cos + cross(k, v)*sin
            
            # Position (Rotate r_vec)
            r_L = r_vec * c + np.cross(z_hat, r_vec) * s
            p_L = p_pri + r_L
            
            # Velocity (Rotate v_vec)
            v_rel_L = v_vec * c + np.cross(z_hat, v_vec) * s
            v_L = v_pri + v_rel_L
            
        else:
            raise ValueError(f"Lagrange point {point_name} not supported")
            
        return np.concatenate([p_L, v_L]).tolist()
