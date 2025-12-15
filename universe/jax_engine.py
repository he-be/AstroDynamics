
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, LinearInterpolation, SaveAt
from typing import Tuple, Optional, Any

from universe.jax_utils import get_gm_values

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

class JAXEngine:
    def __init__(self):
        self.gms = get_gm_values() # [Sun, Jup, Sat, Io, Eur, Gan, Cal]
        self.g0 = 9.80665

    def get_vector_field(self, moon_interp_path: Any, steering_mode: str = 'constant'):
        """
        Returns a JAX-compatible vector field function f(t, y, args).
        
        Args:
            moon_interp_path: A diffrax.Path (LinearInterpolation) yielding 
                              [x_sun, y_sun, z_sun, x_jup, ..., x_cal, y_cal, z_cal] (Total 7 bodies * 3 = 21 coords)
                              relative to SSB or whatever common frame (usually Jup-centered if we optimize relative).
                              Actually, usually we simulate relative to Jupiter.
                              Let's assume the interpolation provides positions of:
                              [Sun, Saturn, Io, Europa, Ganymede, Callisto] relative to Jupiter. (6 bodies)
                              Jupiter itself is at (0,0,0) in Jup-Centered frame.
            steering_mode: 'constant' or 'linear_tangent'
        """
        
        gms = self.gms
        # Indices in GM array: 0:Sun, 1:Jup, 2:Sat, 3:Io, 4:Eur, 5:Gan, 6:Cal
        # We assume simulation is in Jupiter-Centered Inertial Frame.
        
        def dynamics(t, y, args):
            # y: [rx, ry, rz, vx, vy, vz, mass]
            r_ship = y[0:3]
            v_ship = y[3:6]
            mass_ship = y[6]
            
            # 1. Gravity
            # Get body positions at time t
            # interp(t) returns array of shape (6, 3) -> [Sun, Sat, Io, Eur, Gan, Cal]
            # Wait, LinearInterpolation returns flat array if initialized flat.
            # Let's assume input shape is (N_bodies * 3).
            body_pos_flat = moon_interp_path.evaluate(t)
            
            acc_g = jnp.zeros(3)
            
            # Jupiter (Central Body) - Fixed at 0,0,0
            r_jup = jnp.zeros(3)
            d_jup = r_ship - r_jup
            dist_jup = jnp.linalg.norm(d_jup)
            acc_g += -gms[1] * d_jup / (dist_jup**3)
            
            # Other Bodies
            # Mapping: 0->Sun(GM[0]), 1->Sat(GM[2]), 2->Io(GM[3]), 3->Eur(GM[4]), 4->Gan(GM[5]), 5->Cal(GM[6])
            gm_indices = jnp.array([0, 2, 3, 4, 5, 6]) 
            
            # We need to loop or use vectorized ops. 
            # Reshape body_pos to (6, 3)
            others_pos = body_pos_flat.reshape(6, 3)
            
            def body_acc_fn(i, acc_curr):
                pos = others_pos[i]
                gm = gms[gm_indices[i]]
                d = r_ship - pos
                dist = jnp.linalg.norm(d)
                return acc_curr - gm * d / (dist**3)
            
            # Unroll loop for speed (small N=6)
            for i in range(6):
                acc_g = body_acc_fn(i, acc_g)
                
            # 2. Thrust
            # args: params (Control Parameters)
            # constant: [tx, ty, tz, flow_rate]
            # lts: [ax, ay, az, bx, by, bz, flow_rate, thrust_mag]
            
            acc_t = jnp.zeros(3)
            dm = 0.0
            
            if steering_mode == 'constant':
                # args: [ux, uy, uz, flow_rate, thrust_mag]
                u_vec = args[0:3]
                flow_rate = args[3]
                thrust_mag = args[4]
                
                # F = ma -> a = F/m
                # u_vec is Unit Direction
                force_vec = u_vec * thrust_mag
                acc_t = force_vec / (mass_ship * 1000.0) 
                
                dm = -flow_rate
                
            elif steering_mode == 'linear_tangent':
                 # args: [ax, ay, az, bx, by, bz, flow_rate, thrust_mag]
                 a = args[0:3]
                 b = args[3:6]
                 flow_rate = args[6]
                 thrust_mag = args[7]
                 
                 # Control Law: u = unit(a + b*t)
                 # Note: t here is time from start of burn? Or absolute?
                 # Even in LTS, definitions vary. Usually t is relative to t0.
                 # Let's assume t is absolute, but usually 'b*t' assumes t starts at 0.
                 # Optimization params usually assume t_relative.
                 # We'll subtract t0 in the wrapper or params.
                 # For now, let's assume params 'a' accounts for t0 offset (a_new = a + b*t0).
                 
                 target_vec = a + b * t
                 target_norm = jnp.linalg.norm(target_vec)
                 
                 # Singularity protection
                 u_dir = jnp.where(target_norm > 1e-9, target_vec / target_norm, jnp.array([1.0, 0.0, 0.0]))
                 
                 force_vec = u_dir * thrust_mag
                 acc_t = force_vec / (mass_ship * 1000.0)
                 
                 dm = -flow_rate
            
            acc_total = acc_g + acc_t
            
            return jnp.concatenate([v_ship, acc_total, jnp.expand_dims(dm, axis=0)])
            
        return dynamics

    def propagate(self, 
                 state_init, 
                 t_span, 
                 control_params, 
                 moon_interp, 
                 steering_mode='constant',
                 max_steps=5000000,
                 rtol=1e-6,
                 atol=1e-6,
                 throw=True):
        
        t0, t1 = t_span
        
        # Define term immediately
        term = ODETerm(self.get_vector_field(moon_interp, steering_mode))
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=rtol, atol=atol)
        
        sol = diffeqsolve(
            term, solver,
            t0=t0, t1=t1,
            dt0=10.0,
            y0=state_init,
            args=control_params,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=throw
        )
        
        return sol

