import numpy as np
from engine import PhysicsEngine
from optimization import PorkchopOptimizer

def test_optimization_sanity():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    optimizer = PorkchopOptimizer(engine)
    
    # Ganymede to Callisto
    # 2025-01-01
    t_start = "2025-01-01T00:00:00Z"
    
    # Parking Orbits (Low)
    # Ganymede R=2634. LEO ~ 200km alt -> 2834
    r_park_gan = 2834.0
    # Callisto R=2410. LEO ~ 200km alt -> 2610
    r_park_cal = 2610.0
    
    print("Running optimization (Grid Search 10 days)...")
    # Search 10 days window, flight time 1-10 days
    # Step = 1.0 day
    
    best_dv, best_params = optimizer.optimize_window(
        'ganymede', 'callisto',
        t_start, 
        window_duration_days=10.0,
        flight_time_range_days=(1.0, 10.0),
        r_park_dep=r_park_gan,
        r_park_arr=r_park_cal,
        step_days=2.0, # Coarse
        dt_step_days=2.0
    )
    
    print(f"Optimization Result:")
    print(f" Best Delta V: {best_dv:.4f} km/s")
    print(f" Best Launch: {best_params[0]}")
    print(f" Best TOF: {best_params[1]} days")
    
    # Sanity Checks
    # Hohmann estimate Delta V total (from LEO)
    # v_dep_inf ~ ?
    # v_circ_gan ~ 10.8 km/s. v_circ_cal ~ 8.2 km/s.
    # Transfer ~ 3 km/s dv total?
    # Oberth reduces it. Usually ~1-2 km/s for Jup moons.
    # If 10 days isn't enough to find ideal, value might be higher.
    # But shouldn't be Inf.
    
    assert best_dv < 10.0, f"Delta V {best_dv} is too high (Should be < 10 km/s)"
    assert best_dv > 0.1, "Delta V too low"
    
    print("SUCCESS: Optimization returned valid result.")

if __name__ == "__main__":
    test_optimization_sanity()
