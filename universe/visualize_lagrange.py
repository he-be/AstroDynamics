import numpy as np
import matplotlib.pyplot as plt
from engine import PhysicsEngine
from oracle import Oracle
from datetime import datetime, timedelta

def rotate_vector(vec, angle_rad):
    """Rotate 2D vector by angle"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y = vec[0], vec[1]
    return np.array([x*c - y*s, x*s + y*c, vec[2]])

def get_rotating_frame_coordinates(pos_inertial, ganymede_pos_inertial):
    """
    Transform inertial position to rotating frame where Ganymede is fixed on X-axis.
    """
    # Angle of Ganymede
    theta = np.arctan2(ganymede_pos_inertial[1], ganymede_pos_inertial[0])
    
    # Rotate position by -theta
    return rotate_vector(pos_inertial, -theta)

def run_lagrange_test():
    print("Initializing Engine & Oracle...")
    engine = PhysicsEngine()
    oracle = Oracle(engine)
    
    start_time = "2025-01-01T00:00:00Z"
    duration_days = 90 # Need long duration to see libration
    dt = 3600 # 1 hour steps
    total_steps = int(duration_days * 24 * 3600 / dt)
    
    print(f"Simulating {duration_days} days for L4 stability check (Optimized)...")
    
    # 1. Calculate Initial L4 State using Oracle
    l4_pos, l4_vel = oracle.find_l4_state('jupiter', 'ganymede', start_time)
    
    # Particle 1: Optimized L4
    state_l4 = np.concatenate([l4_pos, l4_vel]).tolist()
    
    # Particle 2: Offset (should librate/tadpole)
    # Offset position slightly towards Jupiter (99% radius)
    # We apply the offset vector in the direction of Jupiter (which is roughly -position vector)
    # Actually, simpler to just scale the position vector in Jovicentric frame.
    offset_pos = l4_pos * 0.99
    # Keep the same velocity as the optimized L4 point (or adjust for Keplerian shear? Same vel is fine for small offset)
    state_offset = np.concatenate([offset_pos, l4_vel]).tolist()
    
    # Storage
    traj_l4_rot = {'x': [], 'y': []}
    traj_offset_rot = {'x': [], 'y': []}
    ganymede_rot = {'x': [], 'y': []} 
    
    # Metrics
    l4_deviations = [] # Distance from theoretical L4
    offset_deviations = [] # Distance from theoretical L4 for offset particle
    timestamps = []
    
    # Storage for Plotting (Rotating Frame)
    l4_positions = []
    offset_positions = []
    ganymede_positions = []
    
    current_time_iso = start_time
    
    # Simulation Loop (Single Shot Propagation)
    t_eval = np.arange(0, duration_days * 24 * 3600 + dt, dt)
    
    print("Propagating trajectories...")
    states_l4 = engine.propagate_interpolated(state_l4, start_time, duration_days * 24 * 3600, cache_step=600, t_eval=t_eval)
    states_offset = engine.propagate_interpolated(state_offset, start_time, duration_days * 24 * 3600, cache_step=600, t_eval=t_eval)
    
    print("Analyzing trajectory data...")
    
    from datetime import datetime, timedelta, timezone
    if start_time.endswith('Z'):
        t_str = start_time[:-1] + '+00:00'
    else:
        t_str = start_time
    start_dt_obj = datetime.fromisoformat(t_str)
    
    for i, t_sec in enumerate(t_eval):
        # Current Time
        curr_dt_obj = start_dt_obj + timedelta(seconds=float(t_sec))
        current_time_iso = curr_dt_obj.isoformat().replace('+00:00', 'Z')
        
        # Get Ganymede Position and Velocity
        g_pos_now, g_vel_now = engine.get_body_state('ganymede', current_time_iso)
        
        # Calculate Orbital Normal
        normal = oracle.get_orbital_normal(g_pos_now, g_vel_now)
        
        # Theoretical L4 is 60 degrees ahead in the orbital plane
        l4_theo = oracle.rotate_vector_3d(g_pos_now, normal, np.pi/3)
        
        # L4 Particle State
        s_l4 = states_l4[i]
        pos_l4 = np.array(s_l4[:3])
        
        # Offset Particle State
        s_off = states_offset[i]
        pos_offset = np.array(s_off[:3])
        
        # Deviations
        dev_l4 = np.linalg.norm(pos_l4 - l4_theo)
        l4_deviations.append(dev_l4)
        
        dev_offset = np.linalg.norm(pos_offset - l4_theo)
        offset_deviations.append(dev_offset)
        
        # Rotating Frame Coordinates
        r_hat = g_pos_now / np.linalg.norm(g_pos_now)
        n_hat = normal
        t_hat = np.cross(n_hat, r_hat)
        
        def to_rot_frame(pos_inertial):
            x = np.dot(pos_inertial, r_hat)
            y = np.dot(pos_inertial, t_hat)
            return np.array([x, y])
            
        l4_rot = to_rot_frame(pos_l4)
        offset_rot = to_rot_frame(pos_offset)
        gan_rot = to_rot_frame(g_pos_now)
        
        l4_positions.append(l4_rot)
        offset_positions.append(offset_rot)
        ganymede_positions.append(gan_rot)
        
        timestamps.append(t_sec / (24*3600)) # Days
        
        if i % 100 == 0:
            print(f"Processing Step {i}/{len(t_eval)}", end='\r')
            
    print("\nAnalysis Complete. Plotting...")
    
    # Quantitative Check
    max_deviation = np.max(l4_deviations)
    final_deviation = l4_deviations[-1]
    
    print(f"Max Deviation from Theoretical L4: {max_deviation:.2f} km")
    print(f"Final Deviation from Theoretical L4: {final_deviation:.2f} km")
    
    # Define Stability Threshold (e.g., 10% of Ganymede orbital radius ~1,070,000 km -> 100,000 km)
    # In a real N-body system, L4 is a region, not a point, so some oscillation is expected.
    # But for "Exact L4" injection, it should be relatively small.
    threshold = 50000.0 
    if max_deviation < threshold:
        print(f"PASS: Deviation is within stability threshold ({threshold} km)")
    else:
        print(f"WARNING: Deviation exceeds threshold ({threshold} km)")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.style.use('dark_background')
    
    # Convert to arrays for plotting
    l4_arr = np.array(l4_positions)
    offset_arr = np.array(offset_positions)
    gan_arr = np.array(ganymede_positions)
    
    # Plot 1: Rotating Frame Trajectory
    ax1.scatter([0], [0], color='orange', s=100, label='Jupiter')
    avg_g_x = np.mean(gan_arr[:, 0])
    ax1.scatter([avg_g_x], [0], color='white', s=50, label='Ganymede')
    
    l4_theo_x = avg_g_x * np.cos(np.pi/3)
    l4_theo_y = avg_g_x * np.sin(np.pi/3)
    ax1.scatter([l4_theo_x], [l4_theo_y], marker='x', color='yellow', label='L4 (Theoretical Mean)')
    
    ax1.plot(l4_arr[:, 0], l4_arr[:, 1], color='cyan', linewidth=1, label='Particle @ L4', alpha=0.8)
    ax1.plot(offset_arr[:, 0], offset_arr[:, 1], color='magenta', linewidth=1, label='Particle @ L4 Offset', alpha=0.8)
    
    ax1.set_title(f"Jupiter-Ganymede Rotating Frame ({duration_days} days)")
    ax1.set_xlabel("x' (km)")
    ax1.set_ylabel("y' (km)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.axis('equal')
    
    # Plot 2: Deviation over Time
    ax2.plot(timestamps, l4_deviations, color='cyan', label='Optimized L4')
    ax2.plot(timestamps, offset_deviations, color='magenta', linestyle='--', label='Offset Particle')
    ax2.set_title("Deviation from Theoretical L4 Point (3D)")
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylabel("Distance (km)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.savefig("lagrange_test.png", dpi=150)
    print("Saved lagrange_test.png")

if __name__ == "__main__":
    run_lagrange_test()
