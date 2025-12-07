import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add path to universe
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.scenarios.ganymede_callisto import execute_gan_cal_mission

def test_robustness():
    print("=== Ganymede-Callisto Robustness Test ===")
    engine = PhysicsEngine()
    
    # Test Cases: [Launch Date, Flight Time (Days)]
    # Based on approximate windows or offsets from known optimal.
    # Optimal was 2025-02-24, 6.0 days.
    
    test_cases = [
        ("2025-02-24T00:00:00Z", 6.0, "Optimal"),
        ("2025-02-26T00:00:00Z", 6.0, "Optimal + 2 Days"),
        ("2025-02-22T00:00:00Z", 6.0, "Optimal - 2 Days"),
        # Maybe a different window in Jan?
        # Jan 15? (Guessing logic, porkchop would find it)
        # Let's stick to local robustness around the solution.
    ]
    
    results = []
    
    for t_launch, dt_days, label in test_cases:
        print(f"\n--- Testing Case: {label} ---")
        try:
            dist, _ = execute_gan_cal_mission(engine, t_launch, dt_days)
            results.append((label, dist, "PASS" if dist < 600000 else "FAIL"))
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
            results.append((label, -1, "ERROR"))
            
    print("\n=== Robustness Summary ===")
    print(f"{'Case':<20} | {'Error (km)':<15} | {'Status':<10}")
    print("-" * 50)
    for label, dist, status in results:
        print(f"{label:<20} | {dist:<15.1f} | {status:<10}")
        
    # Check if all passed (ignore ERRORs for threshold check)
    failures = [r for r in results if r[2] != "PASS"]
    if not failures:
        print("\nSUCCESS: All test cases demonstrated consistent arrival accuracy.")
    else:
        print("\nFAILURE: Some test cases deviated significantly.")

if __name__ == "__main__":
    test_robustness()
