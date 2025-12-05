import numpy as np
from planning import solve_lambert

def test_lambert_hohmann():
    """
    Verify Lambert solver against analytical Hohmann transfer.
    """
    # 1. Setup
    # Central body: Earth-like (or Jupiter)
    mu = 3.986004418e5 # Earth KM^3/s^2
    r_earth = 6378.0
    
    # Orbit 1: LEO (r = 7000 km)
    r1_mag = 7000.0
    v1_circ = np.sqrt(mu / r1_mag)
    
    # Orbit 2: GEO (r = 42164 km)
    r2_mag = 42164.0
    v2_circ = np.sqrt(mu / r2_mag)
    
    # 2. Theoretical Hohmann Transfer
    a_trans = (r1_mag + r2_mag) / 2.0
    # Transfer velocity at departure (Perigee of transfer ellipse)
    # Vis-viva: v^2 = mu * (2/r - 1/a)
    v_dep_mag = np.sqrt(mu * (2.0/r1_mag - 1.0/a_trans))
    dv_theo = v_dep_mag - v1_circ
    
    # Time of Flight
    # Half period of transfer ellipse
    tof = np.pi * np.sqrt(a_trans**3 / mu)
    
    print(f"Hohmann Transfer:")
    print(f"R1: {r1_mag} km, R2: {r2_mag} km")
    print(f"TOF: {tof:.4f} s")
    print(f"V_dep: {v_dep_mag:.4f} km/s")
    print(f"DV: {dv_theo:.4f} km/s")
    
    # 3. Lambert Solver Setup
    # r1 vector (on X axis)
    r1 = np.array([r1_mag, 0, 0])
    
    # r2 vector (on -X axis, 180 deg Hohmann)
    # Note: Lambert solver usually struggles with exactly 180 deg (singularity)?
    # Standard solvers handle it or require slight offset.
    # Let's try exactly 180 first. If fails, 179.9.
    r2 = np.array([-r2_mag, 0, 0])
    
    print("\nRunning Lambert Solver...")
    try:
        v1_sol, v2_sol = solve_lambert(r1, r2, tof, mu)
    except Exception as e:
        print(f"Solver Error: {e}")
        # Try offset if singularity
        print("Retrying with slight offset (179 deg)...")
        theta = np.deg2rad(179.0)
        r2 = np.array([r2_mag * np.cos(theta), r2_mag * np.sin(theta), 0])
        # Recompute TOF roughly (Hohmann approx still close)
        # Actually exact TOF changes.
        # But let's check if solver returns SOMETHING reasonable for exact TOF of Hohmann.
        v1_sol, v2_sol = solve_lambert(r1, r2, tof, mu)

    # 4. Analysis
    v1_sol_mag = np.linalg.norm(v1_sol)
    print(f"Solver V1: {v1_sol} (Mag: {v1_sol_mag:.4f})")
    
    # Deviation from Hohmann V_dep
    err_v = abs(v1_sol_mag - v_dep_mag)
    print(f"Error Magnitude: {err_v:.6f} km/s")
    
    assert np.isclose(v1_sol_mag, v_dep_mag, rtol=0.01), \
        f"Lambert result mismatch! Expected {v_dep_mag}, got {v1_sol_mag}"
        
    print("SUCCESS: Lambert solver matches Hohmann transfer.")

if __name__ == "__main__":
    test_lambert_hohmann()
