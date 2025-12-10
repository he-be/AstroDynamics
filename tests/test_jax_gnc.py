
import pytest
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.jax_planning import JAXPlanner

def test_gnc_maneuver_success():
    """
    Verifies that plan_correction_maneuver can fix a trajectory error.
    """
    print("\n[Test] GNC Maneuver Planning...")
    
    engine = PhysicsEngine()
    planner = JAXPlanner(engine)
    
    # 1. Setup Scenario: Simple Drift near Ganymede?
    # Or just Deep Space.
    # Let's use a known state.
    # 1. Setup Scenario with History
    t_start_iso = "2025-06-01T12:00:00Z"
    t_obj = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
    
    # Define Previous State (10 mins before)
    t_prev = t_obj - timedelta(seconds=600.0)
    t_prev_iso = t_prev.isoformat().replace('+00:00', 'Z')
    
    r_prev = [1.0e6, -600.0, 0.0]
    v_prev = [0.0, 1.0, 0.0]
    m0 = 1000.0
    
    prev_state = {
        'position': r_prev, 'velocity': v_prev, 'mass': m0, 'time': t_prev_iso
    }
    
    # Propagate to Current State
    setup_log = planner.evaluate_trajectory(
        r_start=r_prev, v_start=v_prev, t_start_iso=t_prev_iso,
        dt_seconds=600.0, mass=m0, n_steps=10
    )
    current_state = setup_log[-1]
    
    # 2. Define "Target" ...
    dt = 3600.0
    # Update t_obj to actual current time
    t_curr_iso = current_state['time']
    t_curr_obj = datetime.fromisoformat(t_curr_iso.replace('Z', '+00:00'))
    
    t_target = t_curr_obj + timedelta(seconds=dt)
    t_target_iso = t_target.isoformat().replace('+00:00', 'Z')
    
    # Target: Drift + Error
    # From 1e6, coasting 3600s @ 1km/s -> Y increase 3600.
    # We want Y increase 4000.
    # Position: r_curr + [0, 4000, 0]?
    # r_curr is approx [1e6, 0, 0].
    # So Target [1e6, 4000, 0].
    
    target_pos = [current_state['position'][0], current_state['position'][1] + 4000.0, current_state['position'][2]]
    
    # 3. Plan Maneuver
    result = planner.plan_correction_maneuver(
        current_state=current_state,
        target_pos=target_pos,
        target_time_iso=t_target_iso,
        thrust=1000.0,
        isp=300.0,
        tolerance_km=1.0,
        heuristic_offset=True,
        previous_state=prev_state
    )
    
    print(f"  Result Skipped: {result['skipped']}")
    if not result['skipped']:
        print(f"  Delta-V Est: {result.get('delta_v', 0.0)*1000:.1f} m/s")
        print(f"  Final Error: {result['final_error_km']:.4f} km")
        print(f"  Burn Duration: {result['duration']:.2f} s")
    else:
        print(f"  Reason: {result.get('reason', 'Unknown')}")
        
    assert not result['skipped']
    assert result['final_error_km'] < 5.0
    
    # Verify DV magnitude
    # Expect ~110 m/s
    dv = result.get('delta_v', 0.0)
    assert 0.05 < dv < 0.20

def test_gnc_skip_logic():
    """
    Verifies that plan_correction_maneuver SKIPS if error is low.
    """
    print("\n[Test] GNC Skip Logic...")
    engine = PhysicsEngine()
    planner = JAXPlanner(engine)
    
    t_start_iso = "2025-06-01T12:00:00Z"
    r0 = [1.0e6, 0.0, 0.0]
    v0 = [0.0, 1.0, 0.0]
    m0 = 1000.0
    
    current_state = {'position': r0, 'velocity': v0, 'mass': m0, 'time': t_start_iso}
    
    # Target matches natural coast exactly (or close enough)
    # Natural coast ~ [1e6, 3600, 0]
    target_pos = [1.0e6, 3600.0, 0.0] 
    
    dt = 3600.0
    t_obj = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
    t_target_iso = (t_obj + timedelta(seconds=dt)).isoformat().replace('+00:00', 'Z')
    
    result = planner.plan_correction_maneuver(
        current_state=current_state,
        target_pos=target_pos,
        target_time_iso=t_target_iso,
        thrust=1000.0,
        isp=300.0,
        tolerance_km=10.0 # Wide tolerance
    )
    
    print(f"  Result Skipped: {result['skipped']}")
    if result['skipped']:
        print(f"  Reason: {result.get('reason')}")
        
    assert result['skipped'] is True

if __name__ == "__main__":
    test_gnc_maneuver_success()
    test_gnc_skip_logic()
