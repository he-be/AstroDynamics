
import pytest
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner
from universe.transfer import Transfer

def test_transfer_workflow():
    """
    Verifies the Transfer high-level workflow.
    We won't run full optimization to save time, but check state transitions.
    """
    print("\n[Test] Transfer Workflow...")
    
    engine = PhysicsEngine()
    planner = JAXPlanner(engine)
    
    # 1. Init
    transfer = Transfer("ganymede", "callisto", planner)
    assert transfer.origin == "ganymede"
    
    # 2. Window Search
    # Use a knwon window or mocks?
    # We can run a short search.
    start_iso = "2025-07-29T12:00:00Z"
    t_launch = transfer.find_window(start_iso, window_days=2.0, flight_time_days=4.0)
    
    assert transfer.t_launch_iso is not None
    assert transfer.opt_v_dep is not None
    
    # 3. Setup Departure (Without Parking to speed up, or with short parking)
    # Let's do parking with very short wait?
    # Window search might return t_launch = start + 0 if optimal is immediate.
    # Force parking duration check.
    
    # Mocking launch time to be +1 hour from start to ensure parking logic triggers?
    # But find_window returns the OPTIMAL time.
    # Let's trust find_window.
    
    transfer.setup_departure(parking_orbit={'altitude': 500.0})
    assert transfer.launch_state is not None
    assert 'position' in transfer.launch_state
    
    # 4. Execute Departure (Mocking/Simulating)
    # This calls finite burn solver. Expensive?
    # We can use a tiny burn or skip execution if we trust JAXPlanner tests.
    # But integration test should run it.
    
    # Reduce iterations for speed
    # We can inject a mock planner? No, too complex.
    # Let's just run it. It takes ~10s.
    
    transfer.execute_departure(thrust=2000.0, isp=3000.0, initial_mass=1000.0)
    assert len(transfer.logs) >= 2 # Parking + Burn
    assert transfer.burn_end_state is not None
    
    # 5. MCC
    result = transfer.perform_mcc(thrust=2000.0, isp=3000.0, fraction=0.5)
    assert result is not None
    
    print("Transfer Workflow Verified.")

if __name__ == "__main__":
    test_transfer_workflow()
