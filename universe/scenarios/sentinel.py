import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import PhysicsEngine
from mission import MissionPlanner, FlightController
from frames import FrameManager
from planning import solve_lambert
import telemetry

class StationKeepingStrategy:
    def __init__(self, engine, controller, frames, target_spec, tolerance_km=50.0):
        self.engine = engine
        self.controller = controller
        self.frames = frames
        self.target_args = target_spec # (point_name, center_body, secondary_body)
        self.tolerance = tolerance_km
        self.planner = MissionPlanner(engine)
        
    def check_and_correct(self, dt_check_days=1.0):
        """
        Intermittent control logic (Deadband).
        Checks current deviation from target L-point.
        If > tolerance, executes correction burn.
        """
        t_curr = self.controller.time_iso
        
        # 1. Get Target State (L2)
        pt_name, center, sec = self.target_args
        target_state_full = self.frames.get_lagrange_point(pt_name, t_curr, center, sec)
        r_target = np.array(target_state_full[:3])
        v_target = np.array(target_state_full[3:6])
        
        # 2. Get Current Ship State
        r_ship = self.controller.get_position()
        v_ship = self.controller.get_velocity()
        
        # 3. Calculate Error
        err_vec = r_ship - r_target
        err_dist = np.linalg.norm(err_vec)
        
        print(f"[{t_curr}] Drift: {err_dist:.2f} km (Tol: {self.tolerance} km)")
        
        if err_dist > self.tolerance:
            print(f"  -> DEADBAND VIOLATION. Planning Correction...")
            return self.perform_correction(r_ship, r_target, v_ship, t_curr)
        else:
            return False

    def perform_correction(self, r_ship, r_target_curr, v_ship, t_curr):
        # Plan to intercept target L-point at t + dt_correction
        # Correction time horizon: 2 days?
        # If we correct too fast, expensive. Too slow, drift diverge.
        # L2 instability timescale ~ Period/2? (3.5 days). 
        # Let's target 1.0 day intercept to be quick.
        dt_correction_days = 1.0
        dt_sec = dt_correction_days * 86400.0
        
        start_dt = datetime.fromisoformat(t_curr.replace('Z', '+00:00'))
        t_intercept = (start_dt + timedelta(days=dt_correction_days)).isoformat().replace('+00:00', 'Z')
        
        # Target state at intercept time
        pt_name, center, sec = self.target_args
        target_future_full = self.frames.get_lagrange_point(pt_name, t_intercept, center, sec)
        r_target_future = np.array(target_future_full[:3])
        v_target_future = np.array(target_future_full[3:6]) # We will need to match velocity too?
        
        # Lambert to intercept
        # Note: Station Keeping usually targets position AND velocity (Rendezvous).
        # We need to MATCH velocity at arrival to stop drifting.
        # So this is a Rendezvous.
        # Two burns? Or just one to put us on manifold?
        # For deadband, we just kick back towards point.
        # But if we don't match velocity at L2, we will fly through it.
        # Intermittent control typically does:
        # Burn 1: Target L2.
        # Burn 2: Stop at L2.
        # Simulation: We just execute Burn 1 here? The next check will see we are close?
        # If we drift through L2 with high V, next check shows small error, but V is wrong.
        # Eventually Error grows.
        # Better: Single Burn to Match Manifold?
        # Or simply: Execute Lambert Burn 1. 24 hours later, check error (which will be small) and V error?
        # The logic `check_and_correct` is called daily.
        # If we do Burn 1 now, we arrive in 24h.
        # At next check (24h later), we are AT L2. But our Velocity is `v_arr_lambert`.
        # `v_target` is different.
        # We will drift fast.
        # So at Arrival (next step), we might need a "Stop Burn".
        # But `check_and_correct` only sees Position Error.
        # If Position Error is small (0), it does nothing.
        # Then we drift.
        # Correction: We should Execute Burn 1 AND Schedule Burn 2?
        # Or simplify: The "Correction" is a Finite Burn that puts us on a trajectory that stays close?
        # Let's implement full Rendezvous: Burn 1 now. Burn 2 at arrival.
        # This keeps us tight.
        
        v_dep, v_arr = self.planner.calculate_transfer(r_ship, r_target_future, dt_sec, self.engine.GM[center])
        
        # Burn 1
        dv1 = v_dep - v_ship
        self.controller.execute_burn(dv1, 2000.0, 3000.0, label="SK Burn 1")
        
        # We assume we Coast for dt_sec?
        # But the main loop controls coasting.
        # We should tell the main loop we are "Busy doing a maneuver"?
        # Or just execute Burn 1, and return "Correction Active - Wait 1 day".
        
        # Let's return the `dt_correction` so main loop coasts exactly that amount, then we do Burn 2.
        return True, dt_correction_days, v_arr, r_target_future

def run_scenario():
    print("=== US-08: Lagrangian Sentinel (Station Keeping) ===")
    
    engine = PhysicsEngine()
    controller = FlightController(engine)
    frames = FrameManager(engine)
    
    t_start = "2025-01-01T00:00:00Z"
    
    # 1. Initialize at L2
    pt_name = 'L2'
    center = 'jupiter'
    sec = 'ganymede'
    
    print(f"Initializing at {center}-{sec} {pt_name}...")
    l2_state = frames.get_lagrange_point(pt_name, t_start, center, sec)
    
    # Add small error to verify controller trigger
    # +100 km error
    p0 = np.array(l2_state[:3]) + np.array([100.0, 0, 0]) 
    v0 = np.array(l2_state[3:6])
    
    state0 = np.concatenate([p0, v0]).tolist()
    controller.set_initial_state(state0, 1000.0, t_start)
    
    strategy = StationKeepingStrategy(engine, controller, frames, (pt_name, center, sec), tolerance_km=50.0)
    
    # Loop 30 days
    total_days = 30.0
    elapsed = 0.0
    step_days = 1.0
    
    burn_2_pending = False
    burn_2_data = None #(time_due, v_arr_pred, r_target)
    
    while elapsed < total_days:
        # Check Pending Burn 2
        if burn_2_pending:
            # We coasted to the intercept time.
            # Execute Burn 2 (Stop)
            # Match Velocity of L2
            t_curr = controller.time_iso
            l2_future = frames.get_lagrange_point(pt_name, t_curr, center, sec)
            v_l2_true = np.array(l2_future[3:6])
            
            # Current (after coast)
            v_ship = controller.get_velocity()
            
            # Simple DV match
            dv2 = v_l2_true - v_ship
            controller.execute_burn(dv2, 2000.0, 3000.0, label="SK Burn 2 (Stop)")
            
            burn_2_pending = False
            # Resume normal stepping
            continue
            
        # Normal Checks
        # Coast 1 step (simulates passage of time)
        # Check Error
        res = strategy.check_and_correct()
        
        if res:
            # Correction planned
            # res is (True, dt_days, v_arr, r_fut)
            _, dt_burn, v_arr_pred, r_fut = res
            
            # We perform Burn 1 inside check_and_correct.
            # Now we must coast for dt_burn.
            controller.coast(dt_burn * 86400.0)
            elapsed += dt_burn
            
            # Setup Burn 2
            burn_2_pending = True
        else:
            # No correction needed, drift for step
            controller.coast(step_days * 86400.0)
            elapsed += step_days
            
    print("\nMission Complete.")
    
    # Analysis
    # Fuel Usage
    fuel_used = 1000.0 - controller.mass
    print(f"Fuel Consumed: {fuel_used:.2f} kg")
    print(f"Maneuvers Executed: {len(controller.maneuver_log)}")
    
    # Plot (Relative to L2?)
    # Transforming L2 relative motion is tricky.
    # Plot in Rotating Frame.
    # We want to see it staying near L2 ([R_l2, 0])
    
    print("Generating Plot...")
    traj = np.array(controller.trajectory_log) # Inertial
    times = []
    # Reconstruct times? Controller doesn't store time history parallel to traj log.
    # (Design limitation of simple FlightController).
    # We'll plot Inertial for now, or skip 3D plot logic complex.
    # Simpler: Plot Inertial X-Y and Ganymede.
    # L2 spirals with Ganymede.
    
    plt.figure(figsize=(10,10))
    plt.plot(traj[:,0], traj[:,1], label='Ship', lw=0.5)
    
    # Plot Ganymede Orbit
    # Sampling
    gan_x, gan_y = [], []
    start_dt = datetime.fromisoformat(t_start.replace('Z', '+00:00'))
    t_span = np.linspace(0, total_days*86400, 100)
    for t in t_span:
        ti = (start_dt + timedelta(seconds=t)).isoformat().replace('+00:00', 'Z')
        p, _ = engine.get_body_state(sec, ti)
        gan_x.append(p[0])
        gan_y.append(p[1])
    plt.plot(gan_x, gan_y, label='Ganymede', alpha=0.3)
    
    plt.axis('equal')
    plt.legend()
    plt.title(f"US-08 Sentinel: 30 Days Station Keeping")
    plt.savefig('scenario_sentinel.png')
    plt.savefig('scenario_sentinel.png')
    print("Saved scenario_sentinel.png")
    
    # Export Telemetry
    telemetry.export_mission_manifest(controller, 'scenario_sentinel.json', mission_name="Lagrangian Sentinel (Station Keeping)", bodies=['jupiter', 'ganymede'])

if __name__ == "__main__":
    run_scenario()
