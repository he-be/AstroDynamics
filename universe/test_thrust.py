import numpy as np
from engine import PhysicsEngine

def test_thrust_tsiolkovsky():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    # 1. Setup
    t_start = "2025-01-01T00:00:00Z"
    state_0 = [1.06e6, 0, 0, 0, 13.0, 0] 
    
    mass_0 = 1000.0 # kg
    thrust_force = 1000.0 # Newtons (1 kN)
    isp = 300.0 # seconds
    g0 = 9.80665
    duration = 60.0 # seconds
    
    thrust_vector = [0.0, thrust_force, 0.0]
    
    # 2. Reference Run (Coast)
    print("Running Reference Coast...")
    state_coast = engine.propagate(state_0, t_start, duration)
    vel_coast = np.array(state_coast[3:6])
    
    # 3. Controlled Run (Burn)
    print("Running Controlled Burn...")
    # Note: API might changes, but plan is propagate_controlled
    state_burn, final_mass = engine.propagate_controlled(
        state_0, t_start, duration, 
        thrust_vector=thrust_vector, 
        mass=mass_0, 
        isp=isp
    )

    vel_burn = np.array(state_burn[3:6])
    
    # 4. Analysis
    m_dot = thrust_force / (isp * g0)
    expected_mass = mass_0 - m_dot * duration
    print(f"Mass: Initial={mass_0}, Expected={expected_mass:.4f}, Actual={final_mass:.4f}")
    
    if not np.isclose(final_mass, expected_mass, atol=1e-6):
        raise Exception(f"Mass depletion mismatch! Expected {expected_mass}, got {final_mass}")
    
    # Delta V Verification
    dv_theoretical = isp * g0 * np.log(mass_0 / final_mass)
    
    dv_observed_vec = vel_burn - vel_coast
    dv_observed = np.linalg.norm(dv_observed_vec)
    
    # Convert km/s (Sim) to m/s
    dv_observed_m_s = dv_observed * 1000.0
    
    print(f"Delta V (m/s): Theoretical={dv_theoretical:.4f}, Observed={dv_observed_m_s:.4f}")
    
    if not np.isclose(dv_observed_m_s, dv_theoretical, rtol=0.01):
        raise Exception(f"Delta V mismatch! Theoretical {dv_theoretical:.2f}, Observed {dv_observed_m_s:.2f}")
        
    print("SUCCESS: Tsiolkovsky verification passed.")

if __name__ == "__main__":
    test_thrust_tsiolkovsky()
