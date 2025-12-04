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
    
    print(f"Verifying Stability for {duration_days} days...")
    
    for i in range(total_steps):
        # Theoretical L4
        g_pos_now, _ = engine.get_body_state('ganymede', curr_time)
        l4_theo = oracle.rotate_vector(g_pos_now, np.pi/3)
        
        # Deviation Opt
        pos_opt = np.array(state_opt[:3])
        d_opt = np.linalg.norm(pos_opt - l4_theo)
        dev_opt.append(d_opt)
        
        # Deviation Geo
        pos_geo = np.array(state_geo[:3])
        d_geo = np.linalg.norm(pos_geo - l4_theo)
        dev_geo.append(d_geo)
        
        timestamps.append(i * dt / (24*3600))
        
        # Propagate
        state_opt = engine.propagate_interpolated(state_opt, curr_time, dt, cache_step=600)
        state_geo = engine.propagate_interpolated(state_geo, curr_time, dt, cache_step=600)
        
        # Update Time
        t_obj = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
        curr_time = (t_obj + timedelta(seconds=dt)).isoformat()
        
        if i % 100 == 0:
            print(f"Step {i}/{total_steps}", end='\r')
            
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
