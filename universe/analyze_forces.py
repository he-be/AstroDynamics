import numpy as np
from engine import PhysicsEngine

def analyze_forces():
    print("Initializing Engine...")
    engine = PhysicsEngine()
    
    time_iso = "2025-01-01T00:00:00Z"
    
    # Get Bodies
    # Jupiter is at (0,0,0) in our frame
    g_pos, _ = engine.get_body_state('ganymede', time_iso)
    
    # Calculate L4 Position (60 deg lead)
    # Rotate Ganymede position
    angle = np.pi/3
    c, s = np.cos(angle), np.sin(angle)
    l4_pos = np.array([
        g_pos[0]*c - g_pos[1]*s,
        g_pos[0]*s + g_pos[1]*c,
        g_pos[2] # Assume small Z
    ])
    
    print(f"L4 Position: {l4_pos} km")
    
    # Calculate Accelerations
    # a = GM / r^2
    
    forces = {}
    
    # 1. Jupiter
    r_jup = np.linalg.norm(l4_pos)
    a_jup = engine.GM['jupiter'] / (r_jup**2)
    forces['Jupiter'] = a_jup
    
    # 2. Ganymede
    r_gan = np.linalg.norm(l4_pos - g_pos)
    a_gan = engine.GM['ganymede'] / (r_gan**2)
    forces['Ganymede'] = a_gan
    
    # 3. Sun
    # Need Sun position relative to Jupiter
    # s_pos, _ = engine.get_body_state('sun', time_iso) # REMOVED
    # engine.get_body_state only handles moons by default logic?
    # Let's check engine.py.
    # It checks self.moons.get(body_name).
    # Sun is not in self.moons in __init__.
    # But we need it.
    # Let's manually get Sun position using the engine's internal objects.
    
    from datetime import datetime, timezone
    if time_iso.endswith('Z'):
        time_iso = time_iso[:-1] + '+00:00'
    dt_obj = datetime.fromisoformat(time_iso)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    t = engine.ts.from_datetime(dt_obj)
    s_vec = (engine.sun - engine.jupiter).at(t).position.km
    
    # Distance from Sun to L4
    # Sun is at s_vec (relative to Jupiter)
    # L4 is at l4_pos (relative to Jupiter)
    # Vector from Sun to L4 = l4_pos - s_vec
    r_sun_vec = l4_pos - s_vec
    r_sun = np.linalg.norm(r_sun_vec)
    a_sun = engine.GM['sun'] / (r_sun**2)
    forces['Sun'] = a_sun
    
    # 4. Other Moons (Io, Europa, Callisto)
    for moon in ['io', 'europa', 'callisto']:
        m_pos, _ = engine.get_body_state(moon, time_iso)
        r_vec = l4_pos - m_pos
        r_val = np.linalg.norm(r_vec)
        a_val = engine.GM[moon] / (r_val**2)
        forces[moon.capitalize()] = a_val
        
    # Tidal Forces (Differential Gravity)
    # The absolute gravity of the Sun pulls Jupiter and the Particle almost equally.
    # The *perturbing* force is the difference: a_sun_at_L4 - a_sun_at_Jupiter.
    # a_tidal ~ 2 * GM_sun * r_jup / R_sun_jup^3
    
    R_sun_jup = np.linalg.norm(s_vec)
    a_sun_tidal = 2 * engine.GM['sun'] * r_jup / (R_sun_jup**3)
    forces['Sun (Tidal/Perturbing)'] = a_sun_tidal
    
    print(f"\n{'Body':<20} | {'Accel (km/s^2)':<20} | {'Ratio to Jupiter':<20}")
    print("-" * 70)
    
    base_a = forces['Jupiter']
    
    # Sort by magnitude
    sorted_forces = sorted(forces.items(), key=lambda x: x[1], reverse=True)
    
    for name, a in sorted_forces:
        ratio = a / base_a
        print(f"{name:<20} | {a:<20.4e} | {ratio:<20.4e}")
        
    # Conclusion
    print("\nAnalysis:")
    print(f"Ganymede Pull: {forces['Ganymede']:.4e}")
    print(f"Sun Tidal Pull:  {forces['Sun (Tidal/Perturbing)']:.4e}")
    
    if forces['Sun (Tidal/Perturbing)'] > forces['Ganymede']:
        print(">> Sun's Tidal Force is STRONGER than Ganymede's gravity at L4!")
        print(">> This explains why simple L4 stability is disrupted.")
    else:
        print(">> Ganymede's gravity dominates Sun's tidal force.")
        print(f">> Ratio Ganymede/SunTidal: {forces['Ganymede']/forces['Sun (Tidal/Perturbing)']:.2f}")

if __name__ == "__main__":
    analyze_forces()
