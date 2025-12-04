import numpy as np
import matplotlib.pyplot as plt
from engine import PhysicsEngine
from datetime import datetime, timedelta

def run_test():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    start_time = "2150-01-01T00:00:00Z"
    duration_hours = 480
    dt = 3600 # 60ß minutes steps for plotting
    total_steps = int(duration_hours * 3600 / dt)
    
    print(f"Simulating {duration_hours} hours in {dt}s steps...")
    
    # Store trajectories
    bodies = ['jupiter', 'io', 'europa', 'ganymede', 'callisto']
    trajectories = {b: {'x': [], 'y': []} for b in bodies}
    ship_traj = {'x': [], 'y': []}
    
    # Initial Ship State: Low Ganymede Orbit (approximate)
    # Get Ganymede position/velocity
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    
    # Offset by 3000km (Ganymede radius ~2634km)
    # Circular orbit velocity at 3000km: v = sqrt(GM/r) = sqrt(9887.83 / 3000) ~= 1.81 km/s
    # Ganymede velocity + Orbital velocity
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    
    current_state = np.concatenate([ship_pos, ship_vel]).tolist()
    current_time_iso = start_time
    
    # Simulation Loop
    for i in range(total_steps):
        # 1. Record Body Positions
        for b in bodies:
            pos, _ = engine.get_body_state(b, current_time_iso)
            trajectories[b]['x'].append(pos[0])
            trajectories[b]['y'].append(pos[1])
            
        # 2. Record Ship Position
        ship_traj['x'].append(current_state[0])
        ship_traj['y'].append(current_state[1])
        
        # 3. Propagate Ship
        # Note: propagate takes total dt from start, but here we want step-by-step for plotting.
        # However, our engine.propagate integrates from t=0 to dt. 
        # So we can just pass the CURRENT state and dt=step_size.
        # BUT, engine.propagate needs the absolute start time of the integration step to calculate moon positions correctly.
        
        current_state = engine.propagate_interpolated(current_state, current_time_iso, dt, cache_step=300)
        
        # Update time
        # Parse, add dt, format
        # Quick hack for ISO string math
        t_obj = datetime.fromisoformat(current_time_iso.replace('Z', '+00:00'))
        t_new = t_obj + timedelta(seconds=dt)
        current_time_iso = t_new.isoformat()
        
        if i % 10 == 0:
            print(f"Step {i}/{total_steps}", end='\r')
            
    print("\nSimulation Complete. Plotting...")
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    colors = {'jupiter': 'orange', 'io': 'yellow', 'europa': 'cyan', 'ganymede': 'white', 'callisto': 'gray'}
    
    for b in bodies:
        plt.plot(trajectories[b]['x'], trajectories[b]['y'], label=b, color=colors[b], alpha=0.7)
        # Plot end point
        plt.scatter(trajectories[b]['x'][-1], trajectories[b]['y'][-1], color=colors[b])
        
    plt.plot(ship_traj['x'], ship_traj['y'], label='Ship (LGO Start)', color='red', linewidth=1)
    
    plt.title(f"N-Body Simulation Test: {start_time} + {duration_hours}h")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.axis('equal')
    
    output_file = "orbit_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    run_test()
