import heyoka as hy
import numpy as np
import time
import mpl_toolkits.mplot3d # Initialize 3D plotting
import sys
sys.setrecursionlimit(10000)
import matplotlib.pyplot as plt
from engine import PhysicsEngine

def rotate_vector(vec, axis, angle_deg):
    """Rotate vector around axis by angle (degrees) using Rodrigues' formula"""
    theta = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    
    cross_prod = np.cross(axis, vec)
    dot_prod = np.dot(axis, vec)
    
    return (vec * np.cos(theta) + 
            cross_prod * np.sin(theta) + 
            axis * dot_prod * (1 - np.cos(theta)))

def run_demo():
    print("Initializing Heyoka L4 Demo...")
    engine = PhysicsEngine()
    
    # 1. Setup N-Body System
    start_time = "2025-01-01T00:00:00Z"
    
    # Bodies to include in simulation
    # We treat all of them as massive bodies
    body_names = ['jupiter', 'io', 'europa', 'ganymede', 'callisto', 'sun']
    
    # Get initial states manually to ensure consistency
    # We use Solar System Barycenter (SSB) as inertial origin
    from datetime import datetime
    dt_utc = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    t = engine.ts.from_datetime(dt_utc)
    
    # Jupiter Barycenter (5)
    jup_bary = engine.planets['jupiter barycenter']
    s_jb_ssb = jup_bary.at(t)
    p_jb_ssb = s_jb_ssb.position.km
    v_jb_ssb = s_jb_ssb.velocity.km_per_s
    
    # Jupiter Center (599) relative to Jup Bary
    jup_center = engine.jup_moons['jupiter']
    s_jc_jb = jup_center.at(t)
    p_jc_jb = s_jc_jb.position.km
    v_jc_jb = s_jc_jb.velocity.km_per_s
    
    # Jupiter Center (Absolute SSB)
    p_jup = p_jb_ssb + p_jc_jb
    v_jup = v_jb_ssb + v_jc_jb
    
    y0 = [] # State vector [x1, y1, z1, vx1, vy1, vz1, ...]
    gms = []
    
    # Add Jupiter (Body 0)
    y0.extend(p_jup)
    y0.extend(v_jup)
    gms.append(engine.GM['jupiter'])
    
    # Add Moons (Bodies 1-4)
    # They are in 'jup365.bsp' relative to Jupiter Barycenter by default?
    # No, Skyfield 'at()' does the chaining if we ask correctly?
    # But manual chaining is safer given previous issues.
    
    moon_indices = {} # Map name to body index
    
    current_idx = 1
    for name in ['io', 'europa', 'ganymede', 'callisto']:
        body = engine.jup_moons[name]
        s_b_jb = body.at(t)
        p_b_ssb = p_jb_ssb + s_b_jb.position.km
        v_b_ssb = v_jb_ssb + s_b_jb.velocity.km_per_s
        
        y0.extend(p_b_ssb)
        y0.extend(v_b_ssb)
        gms.append(engine.GM[name])
        
        moon_indices[name] = current_idx
        current_idx += 1
        
    # Add Sun (Body 5)
    sun = engine.planets['sun']
    s_sun = sun.at(t)
    y0.extend(s_sun.position.km)
    y0.extend(s_sun.velocity.km_per_s)
    gms.append(engine.GM['sun'])
    sun_idx = current_idx
    current_idx += 1
    
    # 2. Calculate L4 Initial State (Body 6)
    # L4 is 60 degrees ahead of Ganymede
    
    # Ganymede State relative to Jupiter (Jovicentric)
    gan_idx = moon_indices['ganymede']
    # Ganymede is Body `gan_idx`. Jupiter is Body 0.
    # We extract from y0 list
    g_p_ssb = np.array(y0[gan_idx*6 : gan_idx*6+3])
    g_v_ssb = np.array(y0[gan_idx*6+3 : gan_idx*6+6])
    
    p_gan_jup = g_p_ssb - p_jup
    v_gan_jup = g_v_ssb - v_jup
    
    # Orbital Plane Normal
    h = np.cross(p_gan_jup, v_gan_jup)
    
    # Rotate Position and Velocity by +60 degrees (Leading L4)
    p_l4_jup = rotate_vector(p_gan_jup, h, 60.0)
    v_l4_jup = rotate_vector(v_gan_jup, h, 60.0)
    
    # Convert L4 to SSB
    p_l4_ssb = p_jup + p_l4_jup
    v_l4_ssb = v_jup + v_l4_jup
    
    # Add Ship (Body 6)
    y0.extend(p_l4_ssb)
    y0.extend(v_l4_ssb)
    gms.append(0.0) # Massless
    ship_idx = current_idx
    
    # 3. Setup Heyoka
    y0 = [float(x) for x in y0]
    gms = [float(x) for x in gms]
    
    print("Compiling Heyoka Integrator (Fast Mode)...")
    sys = hy.model.nbody(len(gms), masses=gms)
    ta = hy.taylor_adaptive(sys, y0)
    
    # 4. Propagate
    duration_years = 5
    duration_days = duration_years * 365
    duration_sec = duration_days * 24 * 3600
    
    print(f"Propagating for {duration_years} years ({duration_days} days)...")
    
    # Generate grid for plotting (Higher resolution: 1 point per hour)
    # dt_grid = 1 hour
    points_per_day = 24
    t_grid = np.linspace(0, duration_sec, duration_days * points_per_day + 1)
    
    start_clock = time.time()
    res = ta.propagate_grid(t_grid)
    end_clock = time.time()
    
    calc_time = end_clock - start_clock
    print(f"Done in {calc_time:.4f} seconds!")
    print(f"Speed: {duration_sec / calc_time / 86400 / 365:.1f} million years per second (real-time)")
    print(f"Real-Time Factor: {duration_sec / calc_time:.1e} x")
    
    # 5. Coordinate Transformation for Visualization
    times = t_grid
    data = res[5]
    
    # Extract positions
    # Columns:
    # 0-5: Jupiter
    # ...
    # 6*3: Ganymede (idx=3) -> cols 18, 19, 20
    # 6*6: Ship (idx=6) -> cols 36, 37, 38
    
    jup_cols = slice(0, 3)
    gan_cols = slice(gan_idx*6, gan_idx*6+3)
    ship_cols = slice(ship_idx*6, ship_idx*6+3)
    
    pos_jup = data[:, jup_cols]
    pos_gan = data[:, gan_cols]
    pos_ship = data[:, ship_cols]
    
    # Transform to Rotating Frame
    # Center: Jupiter
    # X-axis: Jupiter -> Ganymede
    
    # Relative positions
    r_gan = pos_gan - pos_jup
    r_ship = pos_ship - pos_jup
    
    # Definitions for rotating frame basis vectors
    # u_x = r_gan / |r_gan|
    dist_gan = np.linalg.norm(r_gan, axis=1, keepdims=True)
    u_x = r_gan / dist_gan
    
    # u_z = (r_gan x v_gan) / |...|
    # We need velocity of Ganymede too
    vel_gan = data[:, gan_idx*6+3 : gan_idx*6+6]
    vel_jup = data[:, 0+3 : 0+6]
    v_gan = vel_gan - vel_jup
    
    h_vec = np.cross(r_gan, v_gan)
    h_norm = np.linalg.norm(h_vec, axis=1, keepdims=True)
    u_z = h_vec / h_norm
    
    u_y = np.cross(u_z, u_x)
    
    # Project ship position onto this basis
    x_rot = np.sum(r_ship * u_x, axis=1)
    y_rot = np.sum(r_ship * u_y, axis=1)
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    plt.title(f"L4 Stability Demo (Heyoka) - Zoomed\nDuration: {duration_years} Years (Calc Time: {calc_time:.4f}s)")
    plt.xlabel("Radial Distance (km from Jupiter)")
    plt.ylabel("Tangential Distance (km)")
    
    # L4 theoretical point
    # L4 is at (cos(60)*R, sin(60)*R)
    # R ~ 1.07e6 km
    mean_dist = np.mean(dist_gan)
    l4_x = mean_dist * 0.5            # cos(60)
    l4_y = mean_dist * np.sqrt(3)/2   # sin(60)
    
    plt.plot(x_rot, y_rot, '-', linewidth=1.0, alpha=0.9, color='blue', label='Particle Trajectory')
    
    # Plot L4
    plt.scatter([l4_x], [l4_y], marker='+', s=100, color='red', label='Theoretical L4')
    
    # Auto-scale to trajectory only
    x_min, x_max = np.min(x_rot), np.max(x_rot)
    y_min, y_max = np.min(y_rot), np.max(y_rot)
    
    # Add margin (10%)
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    margin = max(margin_x, margin_y, 1000) # Min 1000km
    
    plt.xlim(x_min - margin, x_max + margin)
    plt.ylim(y_min - margin, y_max + margin)
    
    # Aspect ratio equal
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text info
    plt.figtext(0.15, 0.85, f"Method: Taylor Adaptive (Order 10)\nTol: 1e-10", fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save
    filename = "universe/heyoka_demo.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    run_demo()
