import heyoka as hy
import numpy as np
import time
from engine import PhysicsEngine

def run_heyoka_benchmark():
    print("Initializing Heyoka Benchmark...")
    engine = PhysicsEngine()
    
    # Constants
    gm_jupiter = engine.GM['jupiter']
    gm_sun = engine.GM['sun']
    
    # Initial State (Ganymede + Offset)
    start_time = "2025-01-01T00:00:00Z"
    g_pos, g_vel = engine.get_body_state('ganymede', start_time)
    ship_pos = g_pos + np.array([3000.0, 0.0, 0.0])
    ship_vel = g_vel + np.array([0.0, 1.81, 0.0])
    
    # Heyoka Variables
    # x, y, z, vx, vy, vz
    vars = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    x, y, z, vx, vy, vz = vars
    
    # Equations of Motion
    # Central Body (Jupiter)
    r_sq = x**2 + y**2 + z**2
    r = r_sq**(1/2)
    acc_jup = -gm_jupiter / (r**3) * np.array([x, y, z])
    
    # We need positions of Moons and Sun as functions of time.
    # Heyoka supports time-dependent variables, but for high precision we need Ephemeris.
    # Heyoka has `model.n_body` but that's for self-gravitating systems.
    # Here we have a restricted problem (massless particle).
    # We can use `hy.par` for parameters, but they are constant?
    # No, Heyoka supports `hy.time` dependent expressions?
    # Or we can use Taylor series for the bodies?
    
    # For a fair benchmark against "Interpolated Propagator", we should use Splines?
    # Heyoka doesn't support CubicSpline directly in the expression graph easily (requires C++ callback).
    # BUT, Heyoka is designed for N-body.
    # If we want to test "Heyoka Speed", we should simulate the Moons AS MASSIVE BODIES in the system?
    # That would be a full N-body simulation (6 moons + Sun + Ship).
    # This is different from the "Restricted" simulation we did before.
    # But it's a valid benchmark for "Orbital Calculation".
    
    # Let's set up a full N-body system in Heyoka.
    # Bodies: Jupiter, Io, Europa, Ganymede, Callisto, Sun, Ship.
    # Note: Sun is very far, treating it as N-body might be numerically tricky if we center on Jupiter?
    # Heyoka handles it fine.
    
    print("Setting up N-Body System in Heyoka...")
    
    # 1. Get Initial States for ALL bodies
    bodies = ['jupiter', 'io', 'europa', 'ganymede', 'callisto', 'sun']
    gms = [engine.GM[b] for b in bodies]
    
    # State vector: [x1, y1, z1, vx1, vy1, vz1, x2, ...]
    # We need to get states relative to Solar System Barycenter? 
    # Or we can stick to Jovicentric if we include fictional forces?
    # Heyoka `n_body` integrator works best in Inertial Frame.
    # Let's use Jovicentric but treat Jupiter as moving?
    # No, let's use Solar System Barycenter (SSB) for full N-body.
    
    # Load SSB states
    from datetime import datetime, timezone
    dt_utc = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    t = engine.ts.from_datetime(dt_utc)
    
    # We need to access Skyfield bodies directly for SSB
    # engine.planets is 'de440s.bsp'
    # engine.jup_moons is 'jup365.bsp'
    
    # Manual State Chaining
    # 1. Jupiter Barycenter (5) relative to SSB (0) from de440s
    jup_bary = engine.planets['jupiter barycenter']
    state_jb_ssb = jup_bary.at(t)
    p_jb_ssb = state_jb_ssb.position.km
    v_jb_ssb = state_jb_ssb.velocity.km_per_s
    
    # 2. Jupiter Center (599) relative to Jupiter Barycenter (5) from jup365
    jupiter_center = engine.jup_moons['jupiter']
    state_jc_jb = jupiter_center.at(t) # Should be relative to 5
    p_jc_jb = state_jc_jb.position.km
    v_jc_jb = state_jc_jb.velocity.km_per_s
    
    # Jupiter Center relative to SSB
    p_jup_ssb = p_jb_ssb + p_jc_jb
    v_jup_ssb = v_jb_ssb + v_jc_jb
    
    # 3. Bodies
    bodies_dict = {
        'io': engine.jup_moons['io'],
        'europa': engine.jup_moons['europa'],
        'ganymede': engine.jup_moons['ganymede'],
        'callisto': engine.jup_moons['callisto'],
    }
    
    y0 = []
    
    # Order: Jupiter, Io, Europa, Ganymede, Callisto, Sun
    
    # Jupiter (Body 0)
    y0.extend(p_jup_ssb)
    y0.extend(v_jup_ssb)
    
    # Moons (Bodies 1-4)
    for name in ['io', 'europa', 'ganymede', 'callisto']:
        body = bodies_dict[name]
        # State relative to Jupiter Barycenter (5) (default root of jup365)
        state_b_jb = body.at(t)
        p_b_jb = state_b_jb.position.km
        v_b_jb = state_b_jb.velocity.km_per_s
        
        # State relative to SSB
        p_b_ssb = p_jb_ssb + p_b_jb
        v_b_ssb = v_jb_ssb + v_b_jb
        
        y0.extend(p_b_ssb)
        y0.extend(v_b_ssb)
        
    # Sun (Body 5)
    sun = engine.planets['sun']
    state_sun_ssb = sun.at(t)
    y0.extend(state_sun_ssb.position.km)
    y0.extend(state_sun_ssb.velocity.km_per_s)
    
    # Ship (Body 6)
    # Ship relative to Jupiter Center (p_jup_ssb)
    ship_p_ssb = p_jup_ssb + ship_pos
    ship_v_ssb = v_jup_ssb + ship_vel
    
    y0.extend(ship_p_ssb)
    y0.extend(ship_v_ssb)
    
    # Add Ship Mass (0)
    gms.append(0.0)
    
    # Setup Heyoka Integrator
    # hy.model.n_body(vars, masses)
    # We need to define variables for 7 bodies * 6 vars = 42 vars
    
    sys = hy.model.nbody(len(gms), masses=gms)
    ta = hy.taylor_adaptive(sys, y0)
    
    # Benchmark
    duration_days = 30
    duration_sec = duration_days * 24 * 3600
    
    print(f"Propagating for {duration_days} days (Full N-Body)...")
    
    start = time.time()
    # Propagate to end time
    # Heyoka uses `propagate_for` or `propagate_until`
    # Note: time unit in GM is km^3/s^2, so time is seconds.
    
    # We can just call `propagate_until(duration_sec)`
    out = ta.propagate_until(float(duration_sec))
    
    end = time.time()
    wall_time = end - start
    
    print(f"Heyoka Time: {wall_time:.4f} s")
    print(f"Final Time: {ta.time}")
    
    # Calculate Ship Position relative to Jupiter
    # State vector structure: [x1, y1, z1, vx1, vy1, vz1, ...]
    # Jupiter is index 0. Ship is index 6.
    
    final_state = ta.state
    
    jup_final_p = final_state[0:3]
    ship_final_p = final_state[36:39]
    
    rel_p = ship_final_p - jup_final_p
    print(f"Final Ship Pos (Jovicentric): {rel_p}")
    
    # Compare with Engine (Exact)
    # We need to run engine.propagate to compare?
    # Or just print the value and let user compare with previous benchmark.
    # Previous 30 days Exact was ~45s.
    # Previous 30 days JIT was ~1.3s.
    
    rtf = duration_sec / wall_time
    print(f"RTF: {rtf:.1e} x")

if __name__ == "__main__":
    run_heyoka_benchmark()
