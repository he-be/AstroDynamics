import numpy as np
import matplotlib.pyplot as plt
from engine import PhysicsEngine
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta, timezone

class DiagnosticEngine(PhysicsEngine):
    def propagate_custom(self, state_vector, time_iso, dt, active_bodies=['jupiter', 'ganymede']):
        """
        Custom propagator that only calculates gravity from specified bodies.
        """
        # Setup similar to propagate_interpolated but filtering bodies
        if time_iso.endswith('Z'):
            time_iso = time_iso[:-1] + '+00:00'
        dt_obj = datetime.fromisoformat(time_iso)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            
        t_eval_steps = int(dt / 300) + 2 
        t_eval_seconds = np.linspace(0, dt + 300, t_eval_steps)
        t_list = [dt_obj + timedelta(seconds=s) for s in t_eval_seconds]
        ts_objects = self.ts.from_datetimes(t_list)
        
        body_splines = {}
        
        # Prepare Splines for Active Bodies
        if 'ganymede' in active_bodies:
            vectors = (self.moons['ganymede'] - self.jupiter).at(ts_objects).position.km
            body_splines['ganymede'] = CubicSpline(t_eval_seconds, vectors, axis=1)
            
        if 'sun' in active_bodies:
            vectors = (self.sun - self.jupiter).at(ts_objects).position.km
            body_splines['sun'] = CubicSpline(t_eval_seconds, vectors, axis=1)
            
        # Integration
        y0 = np.array(state_vector)
        t_span = (0, dt)
        
        def equations(t, y):
            rx, ry, rz, vx, vy, vz = y
            r_ship = np.array([rx, ry, rz])
            
            # Jupiter (Always Active as Central Body)
            r_mag = np.linalg.norm(r_ship)
            a = -self.GM['jupiter'] * r_ship / (r_mag**3)
            
            for name, spline in body_splines.items():
                r_body = spline(t)
                r_rel = r_ship - r_body
                dist = np.linalg.norm(r_rel)
                
                gm = self.GM[name] if name in self.GM else self.GM['sun']
                
                a_direct = -gm * r_rel / (dist**3)
                a_indirect = -gm * r_body / (np.linalg.norm(r_body)**3)
                a += (a_direct - a_indirect)
                
            return [vx, vy, vz, a[0], a[1], a[2]]

        sol = solve_ivp(equations, t_span, y0, method='RK45', rtol=1e-6, atol=1e-9)
        return sol.y[:, -1].tolist()

def rotate_vector(vec, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c, vec[2]])

def run_diagnostic():
    print("Initializing Diagnostic Engine...")
    engine = DiagnosticEngine()
    
    start_time = "2025-01-01T00:00:00Z"
    duration_days = 30 # Shorter duration is enough to see divergence rate
    dt = 3600
    total_steps = int(duration_days * 24 * 3600 / dt)
    
    # Initial State (L4 Injection)
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    l4_pos = rotate_vector(g_pos, np.pi/3)
    l4_vel = rotate_vector(g_vel, np.pi/3)
    initial_state = np.concatenate([l4_pos, l4_vel]).tolist()
    
    # Scenarios
    scenarios = {
        'Ideal R3BP (Jup+Gan)': ['jupiter', 'ganymede'],
        'Full N-Body (All)': ['jupiter', 'ganymede', 'sun'] # Ignoring other moons for clarity
    }
    
    results = {k: [] for k in scenarios.keys()}
    timestamps = []
    
    print(f"Running Diagnostic for {duration_days} days...")
    
    for name, bodies in scenarios.items():
        print(f"Simulating: {name}")
        state = list(initial_state)
        curr_time = start_time
        
        for i in range(total_steps):
            # Calculate Deviation
            g_pos_now, _ = engine.get_body_state('ganymede', curr_time)
            l4_theo = rotate_vector(g_pos_now, np.pi/3)
            pos_ship = np.array(state[:3])
            deviation = np.linalg.norm(pos_ship - l4_theo)
            
            if name == list(scenarios.keys())[0]: # Only append time once
                timestamps.append(i * dt / (24*3600))
                
            results[name].append(deviation)
            
            # Propagate
            state = engine.propagate_custom(state, curr_time, dt, active_bodies=bodies)
            
            # Update Time
            t_obj = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
            curr_time = (t_obj + timedelta(seconds=dt)).isoformat()
            
            if i % 100 == 0:
                print(f"Step {i}/{total_steps}", end='\r')
        print()

    # Analysis
    dev_r3bp = results['Ideal R3BP (Jup+Gan)'][-1]
    dev_full = results['Full N-Body (All)'][-1]
    
    print("\nDiagnostic Results (Final Deviation after 30 days):")
    print(f"Ideal R3BP: {dev_r3bp:.2f} km")
    print(f"Full N-Body: {dev_full:.2f} km")
    
    if dev_r3bp > 100000:
        print(">> CONCLUSION: Injection Logic is FLAWED. Even in ideal R3BP, the particle drifts.")
    elif dev_full > 10 * dev_r3bp:
        print(">> CONCLUSION: Solar Perturbation is the DOMINANT cause of instability.")
    else:
        print(">> CONCLUSION: Both factors contribute.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    for name, data in results.items():
        plt.plot(timestamps, data, label=name)
        
    plt.title("L4 Deviation: Ideal vs Full Physics")
    plt.xlabel("Days")
    plt.ylabel("Deviation from Theoretical L4 (km)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("stability_diagnosis.png")
    print("Saved stability_diagnosis.png")

if __name__ == "__main__":
    run_diagnostic()
