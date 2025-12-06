import numpy as np

def stumpff_c(z):
    if z > 0:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z
    elif z < 0:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1) / (-z)
    else:
        return 0.5

def stumpff_s(z):
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / (s**3)
    elif z < 0:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / ((-z)**1.5)
    else:
        return 1.0 / 6.0


def kepler_to_cartesian(a, e, i_rad, Omega_rad, omega_rad, nu_rad, mu):
    """
    Convert Keplerian elements to Cartesian State Vectors.
    
    Args:
        a: Semi-major axis (km)
        e: Eccentricity
        i_rad: Inclination (radians)
        Omega_rad: Longitude of Ascending Node (radians)
        omega_rad: Argument of Periapsis (radians)
        nu_rad: True Anomaly (radians)
        mu: Gravitational parameter
        
    Returns:
        r_vec, v_vec (numpy arrays)
    """
    # 1. Position/Velocity in Perifocal Frame
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu_rad))
    
    # Perifocal coordinates
    # r_orbit = [r cos(nu), r sin(nu), 0]
    # v_orbit = sqrt(mu/p) * [-sin(nu), e + cos(nu), 0]
    
    x_orbit = r * np.cos(nu_rad)
    y_orbit = r * np.sin(nu_rad)
    
    vx_orbit = np.sqrt(mu/p) * (-np.sin(nu_rad))
    vy_orbit = np.sqrt(mu/p) * (e + np.cos(nu_rad))
    
    r_perifocal = np.array([x_orbit, y_orbit, 0.0])
    v_perifocal = np.array([vx_orbit, vy_orbit, 0.0])
    
    # 2. Rotation Matrix to Inertial
    # R3(-Omega) * R1(-i) * R3(-omega)
    # Actually just standard rotation matrix R_perifocal_to_inertial
    
    cO = np.cos(Omega_rad)
    sO = np.sin(Omega_rad)
    cw = np.cos(omega_rad)
    sw = np.sin(omega_rad)
    ci = np.cos(i_rad)
    si = np.sin(i_rad)
    
    # R = [ [cO cw - sO sw ci, -cO sw - sO cw ci,  sO si],
    #       [sO cw + cO sw ci, -sO sw + cO cw ci, -cO si],
    #       [sw si,             cw si,             ci   ] ]
    # Wait, simple matrix multiplication of rotations:
    # R = Rz(Omega) @ Rx(i) @ Rz(omega)
    
    Rz_W = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rx_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    
    R = Rz_W @ Rx_i @ Rz_w
    
    r_vec = R @ r_perifocal
    v_vec = R @ v_perifocal
    
    return r_vec, v_vec

def solve_lambert(r1, r2, dt, mu, cw=False, max_revs=0):
    """
    Solve Lambert's problem using Universal Variables (Bate, Mueller, White).
    
    Args:
        r1: Position vector 1 (numpy array)
        r2: Position vector 2 (numpy array)
        dt: Time of flight (seconds)
        mu: Gravitational parameter
        tm: Transfer method (+1 for short way, -1 for long way). Default short way (tm=1).
        
    Returns:
        v1: Velocity vector at r1
        v2: Velocity vector at r2
    """
    tm = -1 if cw else 1
    
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    cross_r1_r2 = np.cross(r1, r2)
    cross_mag = np.linalg.norm(cross_r1_r2)
    
    # Check for 180 degree transfer (singularity)
    # If cross is near zero and dot is negative -> 180 deg
    dot_prod = np.dot(r1, r2)
    
    if cross_mag < 1e-6 and dot_prod < 0:
        raise ValueError("Lambert Solver: 180 degree transfer singularity. Plane undefined.")
        
    # Calculate delta nu (transfer angle)
    # cos(dnu) = dot / (r1 r2)
    
    # Simple calculation of dnu:
    dnu = np.arccos(np.clip(dot_prod / (r1_mag * r2_mag), -1.0, 1.0))
    
    if tm == -1: # Retrograde / Long way?
        dnu = 2 * np.pi - dnu
        
    # Constants
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(dnu)))
    
    # Iteration for z
    # z = alpha * x^2 (where alpha = 1/a)
    # y(z) = r1 + r2 + A * (z*S(z) - 1) / sqrt(C(z))
    # x = sqrt(y/C(z))
    # dt = (x^3 S(z) + A sqrt(y)) / sqrt(mu)
    
    # Newton-Raphson
    z = 0.0
    ratio = 1.0
    iter_max = 50
    
    points = 0
    while abs(ratio) > 1e-6 and points < iter_max:
        points += 1
        C = stumpff_c(z)
        S = stumpff_s(z)
        
        A_sqrt_C = A * np.sqrt(C) # ?
        
        # Standard Universal Variable form:
        # Eq 5.40 in BMW?
        # A = sin(dnu) * sqrt(r1*r2 / (1-cos(dnu)))
        # y = r1 + r2 + A * (z*S - 1)/sqrt(C)
        
        # Guard for C=0 (z large)
        if C == 0: C = 1e-9 # Should catch via limit
        
        y = r1_mag + r2_mag + A * (z*S - 1)/np.sqrt(C)
        
        if y < 0:
             # Try adjusting z?
             # For hyperbolic, z < 0.
             y = abs(y) # Hack
             
        x = np.sqrt(y/C)
        t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        
        # Derivative dt/dz
        # Complex.
        # Approximation or secant method is easier if derivative is annoying.
        # Let's use Secant or Bisect if simple.
        # But Newton is standard.
        
        # Derivative formulation:
        # dt/dz = ...
        # (See Curtis or BMW)
        if z == 0:
            C_prime = -1/24 # Limit of dC/dz at 0?
            S_prime = -1/120 # Limit
        else:
             C_prime = (1/(2*z)) * (1 - z*S - 2*C)
             S_prime = (1/(2*z)) * (C - 3*S)
             
        # dy/dz = A * (sqrt(C) * (S + z*S_prime) - (z*S - 1)*0.5/sqrt(C)*C_prime) / C
        # Too error prone to write from memory.
        
        # Let's use Simple Bisection/Secant on F(z) = t(z) - dt = 0.
        # Much safer.
        pass
        
    # Re-implement using Izzo-style robust solver or a simpler verified python snippet?
    # I'll implement a verified Stumpff iteration with simple bisect/newton logic, 
    # but to ensure correctness I'll use a very clean formulation.
    
    # Formulation from "Curtis":
    # F(z) = (y/C)^1.5 * S + A * sqrt(y) - sqrt(mu)*dt = 0
    # y = r1 + r2 + A * (z*S - 1)/sqrt(C)
    # This works.
    
    # Newton with numerical derivative?
    
    def time_of_flight(z_in):
        C_val = stumpff_c(z_in)
        S_val = stumpff_s(z_in)
        y_val = r1_mag + r2_mag + A * (z_in * S_val - 1) / np.sqrt(C_val)
        if y_val < 0: return float('inf') # Invalid z
        x_val = np.sqrt(y_val / C_val)
        tof_val = (x_val**3 * S_val + A * np.sqrt(y_val)) / np.sqrt(mu)
        return tof_val

    # Use scipy.optimize.root_scalar if available? 
    # Or simple Newton.
    # User env has scipy? Yes.
    
    from scipy.optimize import root_scalar
    
    try:
        # Good guess for z? 0 (Parabola)
        sol = root_scalar(lambda z_in: time_of_flight(z_in) - dt, x0=0.0, x1=1.0, method='secant', rtol=1e-5)
        z = sol.root
    except ValueError:
        # Fallback to Brent
        sol = root_scalar(lambda z_in: time_of_flight(z_in) - dt, bracket=[-100, 100], method='brentq')
        z = sol.root

    # Reconstruction
    C = stumpff_c(z)
    S = stumpff_s(z)
    y = r1_mag + r2_mag + A * (z*S - 1)/np.sqrt(C)
    
    f = 1 - y/r1_mag
    g = A * np.sqrt(y/mu)
    g_dot = 1 - y/r2_mag
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2
