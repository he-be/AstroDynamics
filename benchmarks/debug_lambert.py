import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from universe.planning import solve_lambert
from universe.engine import PhysicsEngine

def debug_lambert():
    # From Log
    # R_TCM2: [-933728.76860344 1421897.77818577  656379.53970844]
    r1 = np.array([-933728.76860344, 1421897.77818577, 656379.53970844])
    
    # P_CAL_ARR: [-1173088.98837058  1346218.32843435   616933.94713372]
    r2 = np.array([-1173088.98837058, 1346218.32843435, 616933.94713372])
    
    dt = 34560.0 # 9.6 hours
    
    engine = PhysicsEngine() # Loads GM
    mu = engine.GM['jupiter']
    print(f"Mu Jupiter: {mu}")
    
    print(f"R1 Mag: {np.linalg.norm(r1)}")
    print(f"R2 Mag: {np.linalg.norm(r2)}")
    print(f"Dt: {dt}")
    
    dist = np.linalg.norm(r2 - r1)
    print(f"Straight Line Dist: {dist}")
    print(f"Straight Line Vel: {dist/dt} km/s")
    
    print("Solving Lambert...")
    v1, v2 = solve_lambert(r1, r2, dt, mu)
    
    print(f"V1: {v1}")
    print(f"V1 Mag: {np.linalg.norm(v1)}")
    
    print(f"V2: {v2}")
    print(f"V2 Mag: {np.linalg.norm(v2)}")
    
    # Verify by propagation (Kepler)
    # Just integrate? Or check specific energy?
    v_inf_sq = np.linalg.norm(v1)**2 - 2*mu/np.linalg.norm(r1)
    print(f"V_inf_sq: {v_inf_sq} (Energy < 0 is Elliptic)")
    
if __name__ == "__main__":
    debug_lambert()
