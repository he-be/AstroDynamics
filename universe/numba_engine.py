import numpy as np
from numba import njit

@njit(cache=True)
def cubic_spline_eval(t, x, c):
    """
    Evaluate cubic spline at time t.
    
    Args:
        t: Time to evaluate (scalar)
        x: Breakpoints (knots) 1D array (sorted)
        c: Coefficients array of shape (4, n_intervals, n_dims)
           c[0]*dx^3 + c[1]*dx^2 + c[2]*dx + c[3]
           
    Returns:
        Result array of shape (n_dims,)
    """
    # Find interval index
    # x is sorted, so we can use searchsorted logic or simple bisection
    # Since t usually increases monotonically in integration, we could optimize,
    # but binary search is O(log N) which is fast enough.
    
    n = x.shape[0]
    if t <= x[0]:
        idx = 0
    elif t >= x[n-1]:
        idx = n - 2
    else:
        # Binary search
        low = 0
        high = n - 1
        while high - low > 1:
            mid = (low + high) // 2
            if x[mid] <= t:
                low = mid
            else:
                high = mid
        idx = low
        
    # Ensure idx is within bounds of coefficients
    # c has shape (4, n-1, n_dims)
    if idx >= c.shape[1]:
        idx = c.shape[1] - 1
        
    dx = t - x[idx]
    
    # Evaluate polynomial
    # c is (4, n_intervals, n_dims)
    # res = c[0]*dx^3 + c[1]*dx^2 + c[2]*dx + c[3]
    
    n_dims = c.shape[2]
    res = np.empty(n_dims, dtype=np.float64)
    
    for i in range(n_dims):
        res[i] = ((c[0, idx, i] * dx + c[1, idx, i]) * dx + c[2, idx, i]) * dx + c[3, idx, i]
        
    return res

@njit(cache=True)
def equations_of_motion_numba(t, state, gm_jupiter, gm_sun, gm_moons, 
                              spline_x, spline_c_sun, spline_c_moons):
    """
    Calculate derivatives [vx, vy, vz, ax, ay, az].
    
    Args:
        t: Current time
        state: [rx, ry, rz, vx, vy, vz]
        gm_jupiter: GM of Jupiter
        gm_sun: GM of Sun
        gm_moons: Array of GM for moons
        spline_x: Spline knots (shared for all bodies)
        spline_c_sun: Spline coeffs for Sun
        spline_c_moons: List/Tuple of Spline coeffs for moons (N_moons, 4, n_int, 3)
                        Wait, Numba handles tuples of arrays better than list of arrays if shapes differ,
                        but here shapes are likely same. Let's assume passed as a 4D array if possible,
                        or we iterate.
                        Actually, passing a 4D array (N_moons, 4, n_int, 3) is best.
    """
    rx, ry, rz, vx, vy, vz = state
    r_ship = np.array([rx, ry, rz])
    r_mag_sq = rx*rx + ry*ry + rz*rz
    r_mag = np.sqrt(r_mag_sq)
    
    # Jupiter (Central Body)
    # a = -GM * r / r^3
    acc = -gm_jupiter * r_ship / (r_mag_sq * r_mag)
    
    # Sun Perturbation
    r_sun = cubic_spline_eval(t, spline_x, spline_c_sun)
    r_rel_sun = r_ship - r_sun
    dist_sun_sq = np.dot(r_rel_sun, r_rel_sun)
    dist_sun = np.sqrt(dist_sun_sq)
    r_sun_mag_sq = np.dot(r_sun, r_sun)
    r_sun_mag = np.sqrt(r_sun_mag_sq)
    
    a_direct = -gm_sun * r_rel_sun / (dist_sun_sq * dist_sun)
    a_indirect = -gm_sun * r_sun / (r_sun_mag_sq * r_sun_mag)
    acc += (a_direct - a_indirect)
    
    # Moons Perturbation
    # spline_c_moons is (N_moons, 4, n_int, 3)
    n_moons = spline_c_moons.shape[0]
    for i in range(n_moons):
        r_moon = cubic_spline_eval(t, spline_x, spline_c_moons[i])
        r_rel = r_ship - r_moon
        dist_sq = np.dot(r_rel, r_rel)
        dist = np.sqrt(dist_sq)
        r_moon_mag_sq = np.dot(r_moon, r_moon)
        r_moon_mag = np.sqrt(r_moon_mag_sq)
        
        gm = gm_moons[i]
        
        a_direct = -gm * r_rel / (dist_sq * dist)
        a_indirect = -gm * r_moon / (r_moon_mag_sq * r_moon_mag)
        acc += (a_direct - a_indirect)
        
    return np.array([vx, vy, vz, acc[0], acc[1], acc[2]])

@njit(cache=True)
def rk4_step(t, state, dt, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons):
    k1 = equations_of_motion_numba(t, state, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons)
    k2 = equations_of_motion_numba(t + 0.5*dt, state + 0.5*dt*k1, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons)
    k3 = equations_of_motion_numba(t + 0.5*dt, state + 0.5*dt*k2, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons)
    k4 = equations_of_motion_numba(t + dt, state + dt*k3, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@njit(cache=True)
def propagate_numba_loop(state_0, t_start, duration, dt, 
                         gm_jupiter, gm_sun, gm_moons, 
                         spline_x, spline_c_sun, spline_c_moons,
                         t_eval=None):
    """
    Main propagation loop.
    
    Args:
        t_eval: Optional array of times to evaluate. If None, returns final state.
                Must be sorted and within [0, duration].
    """
    t = 0.0
    state = state_0.copy()
    
    # If t_eval is provided, we need to store results
    # We assume t_eval is relative time from t_start (0 to duration)
    if t_eval is not None:
        n_eval = t_eval.shape[0]
        out_states = np.empty((n_eval, 6), dtype=np.float64)
        eval_idx = 0
        
        # Handle t_eval[0] == 0
        if n_eval > 0 and t_eval[0] == 0.0:
            out_states[0] = state
            eval_idx += 1
    
    n_steps = int(duration / dt)
    
    for i in range(n_steps):
        # Check if we need to record state for t_eval
        # This is a simple implementation: we record if the NEXT step passes a t_eval point.
        # For high accuracy, we should step EXACTLY to t_eval.
        # But for visualization, linear interpolation or nearest neighbor might be okay?
        # No, let's do it properly: adjust step size if next t_eval is closer than dt.
        # BUT, to keep it simple and fast for now (fixed step RK4), we will just output 
        # the state at the nearest step or interpolate.
        # Let's stick to fixed step loop for raw speed and just check condition.
        
        # Actually, for "Oracle" precision, we might want adaptive step, but RK4 fixed is robust enough if dt is small.
        # Let's just run fixed steps.
        
        state = rk4_step(t, state, dt, gm_jupiter, gm_sun, gm_moons, spline_x, spline_c_sun, spline_c_moons)
        t += dt
        
        if t_eval is not None:
            while eval_idx < n_eval and t_eval[eval_idx] <= t:
                # Simple interpolation or just take current state?
                # Taking current state introduces error up to dt * v.
                # Let's do a linear interpolation between prev_state and state?
                # We don't have prev_state easily unless we store it.
                # For now, just store current state (nearest). 
                # User can use smaller dt for better resolution.
                out_states[eval_idx] = state
                eval_idx += 1
                
    if t_eval is not None:
        return out_states
    else:
        # Return 2D array (1, 6) to match shape
        res = np.empty((1, 6), dtype=np.float64)
        res[0] = state
        return res
