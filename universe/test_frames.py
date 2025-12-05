import numpy as np
from engine import PhysicsEngine
from frames import FrameManager

def test_rotating_frame_l4():
    """
    Verify transformation to Jupiter-Ganymede Rotating Frame.
    A point at theoretical L4 should have near-zero velocity in the rotating frame.
    """
    print("Initializing Engine...")
    engine = PhysicsEngine()
    frames = FrameManager(engine)
    
    t_start = "2025-01-01T00:00:00Z"
    
    # 1. Get Ganymede State (Jovicentric Inertial)
    p_gan, v_gan = engine.get_body_state('ganymede', t_start)
    p_gan = np.array(p_gan)
    v_gan = np.array(v_gan)
    
    # 2. Define Theoretical L4 (Inertial)
    # L4 is 60 degrees ahead of Ganymede in orbital plane.
    # Calculate Normal
    h = np.cross(p_gan, v_gan)
    h_hat = h / np.linalg.norm(h)
    
    # Rotate Position 60 deg
    theta = np.deg2rad(60.0)
    # Rodrigues rotation
    def rotate(v, k, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
    
    p_l4 = rotate(p_gan, h_hat, theta)
    
    # Rotate Velocity 60 deg? 
    # Velocity direction rotates, magnitude same?
    v_l4 = rotate(v_gan, h_hat, theta)
    
    # 3. Transform to Rotating Frame
    # Input: State [rx,ry,rz, vx,vy,vz]
    state_inertial = np.concatenate([p_l4, v_l4])
    
    # Transform
    # generic method: transform(state, time, frame_type, primary, secondary)
    state_rot = frames.transform_to_rotating(
        state_inertial, 
        t_start, 
        center_body='jupiter', 
        secondary_body='ganymede'
    )
    
    p_rot = state_rot[0:3]
    v_rot = state_rot[3:6]
    
    print(f"L4 Inertial Pos: {p_l4}")
    print(f"L4 Inertial Vel: {v_l4}")
    print(f"L4 Rotating Pos: {p_rot}")
    print(f"L4 Rotating Vel: {v_rot}")
    
    # 4. Assertions
    # Position: Should be at distance R from Center, at angle 60 deg?
    # In Rotating Frame:
    # X axis usually aligns with Secondary.
    # So Ganymede is at [R, 0, 0].
    # L4 is at [R * cos(60), R * sin(60), 0] = [R/2, R*sqrt(3)/2, 0].
    r_mag = np.linalg.norm(p_gan)
    expected_pos = np.array([r_mag * np.cos(theta), r_mag * np.sin(theta), 0])
    
    # Note: Frame definition might put X towards secondary.
    # We need to verify implementation matches expectation.
    # Let's assume standard: X defines Primary-Secondary line. Z is normal.
    
    # Deviation in position
    pos_err = np.linalg.norm(p_rot - expected_pos)
    print(f"Position Error: {pos_err:.4f} km")
    
    # There will be some error because 'ganymede' orbit is not perfectly circular,
    # and 'L4' definition used above assumes circular rotation of velocity vector.
    # But it should be reasonably close.
    assert pos_err < 1000.0, f"L4 Position mismatch in rotating frame. Err: {pos_err}"
    
    # Velocity: Should be near zero
    # Because L4 co-rotates.
    vel_mag = np.linalg.norm(v_rot)
    print(f"Velocity Magnitude in Rotating Frame: {vel_mag:.4f} km/s")
    
    # Tolerance: 
    # Ganymede e=0.0013. v ~ 10 km/s.
    # Variation due to eccentricity might be 0.01 - 0.1 km/s.
    assert vel_mag < 0.5, f"L4 should be stationary in rotating frame. Vel: {vel_mag}"
    
    print("SUCCESS: Rotating frame transformation verifies L4 stability.")

if __name__ == "__main__":
    test_rotating_frame_l4()
