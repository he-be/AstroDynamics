import numpy as np
import matplotlib.pyplot as plt
from engine import PhysicsEngine
from oracle import Oracle
from datetime import datetime, timedelta

def run_solver_test():
    print("Initializing Engine & Oracle...")
    engine = PhysicsEngine()
    oracle = Oracle(engine)
    
    start_time = "2025-01-01T00:00:00Z"
    
    # Solve
    print("Running L4 Solver (6-DOF)...")
    l4_pos_opt, l4_vel_opt = oracle.find_l4_state('jupiter', 'ganymede', start_time)
    
    # Geometric Guess (for comparison)
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    l4_pos_geo = oracle.rotate_vector(g_pos, np.pi/3)
    l4_vel_geo = oracle.rotate_vector(g_vel, np.pi/3)
    
    # Propagate Comparison (30 Days)
    duration_days = 30
    dt = 3600
    total_steps = int(duration_days * 24 * 3600 / dt)
    
    state_opt = np.concatenate([l4_pos_opt, l4_vel_opt]).tolist()
    state_geo = np.concatenate([l4_pos_geo, l4_vel_geo]).tolist()
    
    dev_opt = []
    dev_geo = []
    timestamps = []
    
    curr_time = start_time
    
    print(f"Verifying Stability for {duration_days} days (High Performance Run)...")
    
    # Batch Propagation
    t_eval = np.arange(0, duration_days * 24 * 3600, dt) + dt # Exclude 0, steps of dt
    # If 0 is included, it's fine, but verifying usually implies checking FUTURE.
    # range(total_steps) usually implies 0 to N-1?
    # Old logic: Loop i from 0 to total_steps-1.
    # At i=0, curr_time = start_time.
    # Propagate step (0 -> dt).
    # Then loop i=1.
    # So we want state at t=0, t=dt, t=2dt...
    # But propagate_interpolated was updating `state_opt`.
    # At start of loop: check dev at t=0. Propagate to t=dt.
    # So we need output at t=0, dt, 2dt...
    # t_eval should include 0 if we want initial deviation.
    # But engine.propagate(t_eval=[0]) returns state at 0 (y0).
    # Let's use t_eval = [0, dt, 2dt ...].
    t_eval = np.arange(0, duration_days * 24 * 3600, dt)
    
    # Propagate (Heyoka uses reused integrator)
    print("Propagating Optimized Trajectory...")
    states_opt_list = engine.propagate(state_opt, start_time, duration_days*24*3600, t_eval=t_eval)
    
    print("Propagating Geometric Trajectory...")
    states_geo_list = engine.propagate(state_geo, start_time, duration_days*24*3600, t_eval=t_eval)
    
    # Analyze
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    
    for i, t_sec in enumerate(t_eval):
        current_iso = (start_dt + timedelta(seconds=t_sec)).isoformat()
        
        # Theoretical L4
        g_pos_now, _ = engine.get_body_state('ganymede', current_iso)
        l4_theo = oracle.rotate_vector(g_pos_now, np.pi/3)
        
        # Deviation Opt
        s_opt = states_opt_list[i]
        pos_opt = np.array(s_opt[:3])
        d_opt = np.linalg.norm(pos_opt - l4_theo)
        dev_opt.append(d_opt)
        
        # Deviation Geo
        s_geo = states_geo_list[i]
        pos_geo = np.array(s_geo[:3])
        d_geo = np.linalg.norm(pos_geo - l4_theo)
        dev_geo.append(d_geo)
        
        timestamps.append(t_sec / (24*3600))
        
        if i % 100 == 0:
            print(f"Analyzing Step {i}/{len(t_eval)}", end='\r')
            
    print("\nVerification Complete.")
    print(f"Final Deviation (Geometric): {dev_geo[-1]:.2f} km")
    print(f"Final Deviation (Optimized): {dev_opt[-1]:.2f} km")
    
    improvement = dev_geo[-1] / dev_opt[-1] if dev_opt[-1] > 0 else 0
    print(f"Improvement Factor: {improvement:.1f}x")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.plot(timestamps, dev_geo, label='Geometric Guess', color='gray', linestyle='--')
    plt.plot(timestamps, dev_opt, label='Optimized Solver', color='cyan')
    
    plt.title("L4 Stability: Geometric vs Optimized Injection")
    plt.xlabel("Days")
    plt.ylabel("Deviation from Theoretical L4 (km)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("solver_test.png")
    print("Saved solver_test.png")

if __name__ == "__main__":
    run_solver_test()
