
import time
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt

# Enable 64-bit precision (Essential for orbital mechanics)
jax.config.update("jax_enable_x64", True)

def test_diffrax_simple():
    print("Initializing JAX/Diffrax...")
    
    # 1. Define Dynamics (Harmonic Oscillator to start simple, or Kepler)
    # Let's do a simple Kepler 2-body: Earth-Sat
    mu = 398600.4418
    
    def vector_field(t, y, args):
        # y = [rx, ry, rz, vx, vy, vz]
        r = y[0:3]
        v = y[3:6]
        
        r_norm = jnp.linalg.norm(r)
        acc = -mu * r / (r_norm**3)
        
        return jnp.concatenate([v, acc])
    
    term = ODETerm(vector_field)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)
    
    y0 = jnp.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0]) # Approx LEO
    t0 = 0.0
    t1 = 3600.0 * 24.0 # 1 day
    
    print("Compiling & Running (First Run)...")
    t_start = time.time()
    
    # JIT Compile and Run
    @jax.jit
    def run_propagate():
        return diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=0.1, y0=y0,
            stepsize_controller=stepsize_controller,
            max_steps=1000000 
        )
        
    sol = run_propagate()
    sol.ys # Force computation
    jax.block_until_ready(sol.ys)
    
    t_end = time.time()
    print(f"First Run Time (Compile + Exec): {t_end - t_start:.4f} s")
    print(f"Final State: {sol.ys[-1]}")
    
    print("Running (Second Run)...")
    t_start = time.time()
    sol2 = run_propagate()
    jax.block_until_ready(sol2.ys)
    t_end = time.time()
    print(f"Second Run Time (Exec only): {t_end - t_start:.4f} s")
    
    # Check Auto-Diff
    print("\nChecking Auto-Diff...")
    
    @jax.jit
    def get_final_pos_x(y_init):
         sol = diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=0.1, y0=y_init,
            stepsize_controller=stepsize_controller,
            max_steps=1000000 
        )
         return sol.ys[-1][0]
     
    grad_fn = jax.grad(get_final_pos_x)
    
    t_start = time.time()
    g = grad_fn(y0)
    jax.block_until_ready(g)
    t_end = time.time()
    print(f"Gradient Calculation Time: {t_end - t_start:.4f} s")
    print(f"Gradient: {g}")

if __name__ == "__main__":
    test_diffrax_simple()
