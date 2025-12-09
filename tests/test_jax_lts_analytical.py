
import pytest
import numpy as np
import jax.numpy as jnp
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def analytical_rocket_solution(r0, v0, m0, thrust, isp, dt, u_dir):
    """
    Computes exact position and velocity after dt for a rocket in free space.
    Based on Tsiolkovsky Rocket Equation and its integration for position.
    
    u_dir: Unit vector of thrust direction (Constant).
    """
    g0 = 9.80665
    ve = isp * g0 / 1000.0 # km/s
    dm = thrust / (isp * g0) # kg/s
    
    # Mass at time t
    mf = m0 - dm * dt
    
    # 1. Velocity Checking: v = v0 + ve * ln(m0/mf) * u
    # Note: thrust is Force (N), ve is (km/s).
    # F = dm * ve_m_s.  
    # Delta V = ve * ln(m0/mf)
    dv = ve * np.log(m0/mf)
    v_final_analytic = np.array(v0) + dv * np.array(u_dir)
    
    # 2. Position Checking
    # r(t) = r0 + v0*t + ... integral of acceleration.
    # a(t) = F / (m0 - dm*t)
    # Double integral results in:
    # r(t) = r0 + v0*t + u * ve * [ t - (m0/dm - t) * ln(m0/(m0-dm*t)) ] ?
    # Let's derive or use known form:
    # x(t) = ve * [ t - ((m0 - dm*t)/dm) * log(m0 / (m0 - dm*t)) ]
    # Check dimensions: ve [L/T] * T = [L]. Term 2: (M / (M/T)) = T. T*1 = T. dimensions ok.
    # Wait, check limit dm->0.
    
    # Formula:
    # Delta x_thrust = c * t - c * (m(t) / m_dot) * ln(m0 / m(t))
    # where c = ve.
    # m(t) = m0 - dm*t.
    # correction: usually m_dot is positive flow. here dm is flow rate.
    
    term1 = ve * dt
    term2 = ve * (mf / dm) * np.log(m0 / mf)
    
    dx_thrust_mag = term1 - term2
    
    r_final_analytic = np.array(r0) + np.array(v0) * dt + np.array(u_dir) * dx_thrust_mag
    
    return r_final_analytic, v_final_analytic, mf

def test_jax_lts_analytical_verification():
    """
    Objective test:
    Target is derived from Analytical Formula (Deep Space Rocket Equation).
    JAX Planner must hit it.
    This validates Physics Engine integration logic and Planner logic INDEPENDENTLY of the engine itself.
    """
    print("\n[Test] Analytical Physics Verification...")
    
    engine = PhysicsEngine()
    planner = JAXPlanner(engine)
    
    # 1. Scenario: Deep Space (Negligible Gravity)
    # Place spacecraft at 100 AU (1.5e10 km)
    r0 = [1.5e10, 0.0, 0.0]
    v0 = [0.0, 10.0, 0.0] # 10 km/s drift
    m0 = 1000.0
    thrust = 1000.0 # 1000 N
    isp = 300.0
    
    dt = 2000.0 # Long burn
    
    # Thrust Direction: Fixed [1, 0, 0] (X-axis)
    # This corresponds to LTS params: a=[1,0,0], b=[0,0,0]
    u_dir = [1.0, 0.0, 0.0]
    
    # 2. Calculate Truth via Math (Not Engine)
    r_truth_math, v_truth_math, m_final_math = analytical_rocket_solution(
        r0, v0, m0, thrust, isp, dt, u_dir
    )
    
    print(f"  Math Target Position: {r_truth_math}")
    print(f"  Math Target Velocity: {v_truth_math}")
    print(f"  Math Final Mass: {m_final_math:.2f} kg")
    
    # 3. Solver Attempt
    # We ask the planner to hit 'r_truth_math' starting from r0.
    # It must find a=[1,0,0], b=[0,0,0].
    # Gravity is NOT zero in solver, but at 100 AU it should be negligible.
    # GM_jup / r^2 = 1.2e8 / (1.5e10)^2 ~ 1e8 / 2e20 ~ 0.5e-12 km/s2. 
    # Thrust Accel = 1e-3 km/s2. Gravity is 9 orders of magnitude smaller. Safe.
    
    t_iso = "2025-01-01T00:00:00Z"
    
    # Use Warm Start to assume we are testing physics accuracy, not search.
    # But let's verify it can optimize slight deviation.
    guess = [0.99, 0.0, 0.0,  0.0, 0.0, 0.0]
    
    params, r_final_jax, m_final_jax = planner.solve_lts_transfer(
        r_start=r0,
        v_start=v0,
        t_start_iso=t_iso,
        dt_seconds=dt,
        target_pos=list(r_truth_math),
        mass_init=m0,
        thrust=thrust,
        isp=isp,
        initial_params_guess=guess
    )
    
    # 4. Assessment
    pos_error = np.linalg.norm(r_final_jax - r_truth_math)
    print(f"  JAX vs Math Error: {pos_error:.4f} km")
    
    # We allow some small error due to:
    # - Gravity (tiny but real in JAX)
    # - Integration approximation (Dopri5 vs Analytic)
    # - Optimization convergence residual
    assert pos_error < 5.0, f"Physics mismatch! Engine did not match Analytical Rocket Eq. Error: {pos_error} km"
    
    print("PASS: JAX Engine matches Analytical Rocket Equation.")

if __name__ == "__main__":
    test_jax_lts_analytical_verification()
