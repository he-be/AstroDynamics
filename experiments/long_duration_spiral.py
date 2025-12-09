
import time
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def test_spiral_orbit():
    print("=== Long-Duration Low-Thrust Spiral Test (JAX/Diffrax) ===")
    
    # Constants
    mu = 398600.4418 # Earth
    # Low Thrust parameters
    thrust = 1.0 # Newtons
    mass_init = 1000.0 # kg
    isp = 3000.0
    g0 = 9.80665
    flow_rate = thrust / (isp * g0)
    
    # Dynamics with Constant Tangential Thrust
    def vector_field(t, y, args):
        r = y[0:3]
        v = y[3:6]
        m = y[6]
        
        r_norm = jnp.linalg.norm(r)
        v_norm = jnp.linalg.norm(v)
        
        # Gravity
        acc_g = -mu * r / (r_norm**3)
        
        # Thrust (Tangential: along velocity vector)
        u_dir = v / v_norm
        acc_t = (thrust / m) * u_dir # Correct mass division (Newton 1N / 1000kg = 0.001 m/s^2)
        
        acc_total = acc_g + acc_t
        
        # Mass depletion
        dm = -flow_rate
        
        return jnp.concatenate([v, acc_total, jnp.array([dm])])
    
    term = ODETerm(vector_field)
    solver = Dopri5()
    # Adaptive step size
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)
    
    # Initial State (LEO)
    # r = 7000km, v_circ = sqrt(mu/r)
    r_mag = 7000.0
    v_mag = jnp.sqrt(mu / r_mag)
    y0 = jnp.array([r_mag, 0.0, 0.0, 0.0, v_mag, 0.0, mass_init])
    
    t0 = 0.0
    # Simulate for 30 days (very long spiral)
    # 30 * 24 * 3600 = 2,592,000 seconds
    t_end = 30.0 * 24.0 * 3600.0 
    
    print(f"Propagating for {t_end/86400:.1f} days...")
    
    # Compilation
    print("Compiling...")
    t_start = time.time()
    
    @jax.jit
    def run_propagate():
        return diffeqsolve(
            term, solver, t0=t0, t1=t_end, dt0=0.1, y0=y0,
            stepsize_controller=stepsize_controller,
            max_steps=10000000 # Allow many steps
        )
    
    sol = run_propagate()
    jax.block_until_ready(sol.ys)
    t_comp_end = time.time()
    print(f"First Run (Compile+Exec): {t_comp_end - t_start:.4f} s")
    
    # Execution Speed Test
    print("Executing (Warm)...")
    t_start = time.time()
    sol2 = run_propagate()
    jax.block_until_ready(sol2.ys)
    t_exec_end = time.time()
    
    final_r = jnp.linalg.norm(sol2.ys[-1][0:3])
    final_v = jnp.linalg.norm(sol2.ys[-1][3:6])
    final_mass = sol2.ys[-1][6]
    
    print(f"Execution Time: {t_exec_end - t_start:.4f} s")
    print(f"Final Radius: {final_r:.2f} km (Start: {r_mag:.2f} km)")
    print(f"Final Mass: {final_mass:.2f} kg")
    print(f"Altitude Gain: {final_r - r_mag:.2f} km")
    
    # Verify Performance
    if (t_exec_end - t_start) < 2.0:
        print("SUCCESS: Fast integration achieved.")
    else:
        print("WARNING: Integration slow.")

if __name__ == "__main__":
    test_spiral_orbit()
