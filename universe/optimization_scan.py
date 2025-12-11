
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universe.engine import PhysicsEngine
from universe.planning import solve_lambert

def run_optimization_scan():
    print("=== Coarse Grid Search Optimization (Patched Conics) ===")
    
    # 1. Setup
    engine = PhysicsEngine()
    
    # Scan Parameters
    t_start_iso = "2025-07-29T12:00:00Z" # Original Start
    scan_duration_days = 30.0 # Departure Window
    tof_min_days = 2.0
    tof_max_days = 60.0
    
    # Grid Resolution
    # Departure steps: every 6 hours
    dep_steps = int(scan_duration_days * 4) 
    # TOF steps: every 6 hours
    tof_steps = int((tof_max_days - tof_min_days) * 4)
    
    t_dep_vals = np.linspace(0, scan_duration_days, dep_steps)
    tof_vals = np.linspace(tof_min_days, tof_max_days, tof_steps)
    
    print(f"Scanning {dep_steps} Departure Dates x {tof_steps} Flight Times = {dep_steps*tof_steps} cases.")
    
    # Results Storage
    results = [] # (t_dep_iso, tof_days, dv_dep, dv_cap, dv_total)
    
    # Pre-calculate Body States?
    # Engine uses Skyfield, which is fast, but looping is safer.
    
    # Base Time
    dt_base = datetime.fromisoformat(t_start_iso.replace('Z', '+00:00'))
    
    best_total_dv = float('inf')
    best_case = None
    
    for i, d_dep in enumerate(t_dep_vals):
        t_dep_obj = dt_base + timedelta(days=d_dep)
        t_dep_str = t_dep_obj.isoformat().replace('+00:00', 'Z')
        
        # Get Ganymede State
        p_gan, v_gan = engine.get_body_state('ganymede', t_dep_str)
        r1 = np.array(p_gan)
        v1 = np.array(v_gan)
        
        for j, d_tof in enumerate(tof_vals):
            t_arr_obj = t_dep_obj + timedelta(days=d_tof)
            t_arr_str = t_arr_obj.isoformat().replace('+00:00', 'Z')
            
            # Get Callisto State
            p_cal, v_cal = engine.get_body_state('callisto', t_arr_str)
            r2 = np.array(p_cal)
            v2 = np.array(v_cal)
            
            dt_sec = d_tof * 86400.0
            mu_jup = engine.GM['jupiter']
            
            # Solve Lambert
            # Try both prograde (tm=1) and retrograde? Usually prograde.
            # solve_lambert returns v_dep_L, v_arr_L
            try:
                # We need to handle multi-rev? 
                # solve_lambert supports 'max_revs'.
                # For 60 days, Callisto period is 16 days. So ~3-4 revs possible.
                # Standard solve_lambert might be 0-rev only?
                # Let's check planning.py implementation.
                # Assuming 0-rev for now, but if TOF > 1/2 period, Lambert solvers can be tricky.
                # For long transfers, we might need multi-rev Lambert.
                # Checking `planning.py`... standard Izzo/Gooding usually supports it.
                # Calling with max_revs=0 (default).
                
                v_dep_L, v_arr_L = solve_lambert(r1, r2, dt_sec, mu_jup, cw=False, max_revs=0)
                
                # Delta V Calculation
                dv_dep_vec = v_dep_L - v1
                dv_dep = np.linalg.norm(dv_dep_vec)
                
                v_inf_vec = v_arr_L - v2
                v_inf = np.linalg.norm(v_inf_vec)
                
                # Capture DV (Impulsive at 500km altitude)
                # Vis-viva: v^2 = mu(2/r - 1/a). 
                # Hyperbolic arrival (a < 0): v_p^2 = v_inf^2 + 2*mu/r_p
                # Circular orbit (a = r): v_c^2 = mu/r_p
                # dv_cap = |v_p - v_c|
                
                R_cal = 2410.3
                r_p = R_cal + 500.0
                mu_cal = engine.GM['callisto']
                
                v_p = np.sqrt(v_inf**2 + 2*mu_cal/r_p)
                v_c = np.sqrt(mu_cal/r_p)
                dv_cap = abs(v_p - v_c)
                
                total_dv = dv_dep + dv_cap
                
                results.append((d_dep, d_tof, dv_dep, dv_cap, total_dv))
                
                if total_dv < best_total_dv:
                    best_total_dv = total_dv
                    best_case = {
                        't_dep': t_dep_str,
                        'tof': d_tof,
                        'dv_dep': dv_dep,
                        'dv_cap': dv_cap,
                        'total': total_dv,
                        'v_inf': v_inf
                    }
                    
            except Exception:
                # Lambert failure (geometry constraint)
                continue
                
        if i % 10 == 0:
            print(f"Processed Departure Day {d_dep:.1f}...")

    # Report
    print("\n=== Optimization Results ===")
    if best_case:
        print(f"Best Solution Found:")
        print(f"  Departure : {best_case['t_dep']}")
        print(f"  Flight Time: {best_case['tof']:.2f} days")
        print(f"  Dep DV     : {best_case['dv_dep']*1000:.1f} m/s")
        print(f"  Cap DV     : {best_case['dv_cap']*1000:.1f} m/s")
        print(f"  Total DV   : {best_case['total']*1000:.1f} m/s")
        print(f"  V_inf      : {best_case['v_inf']*1000:.1f} m/s")
    else:
        print("No valid solution found.")

    # Visualization (Porkchop)
    # Convert to arrays
    data = np.array(results) 
    # data columns: 0:dep_day, 1:tof, 2:dv_dep, 3:dv_cap, 4:total
    
    # We might want to plot this later. Save to file?
    # For now just print top 5.
    
    # Sort by total DV
    sorted_indices = np.argsort(data[:, 4])
    print("\nTop 5 Candidates:")
    for k in range(5):
        idx = sorted_indices[k]
        row = data[idx]
        t_dep_k = dt_base + timedelta(days=row[0])
        print(f"  {k+1}. [Dep {t_dep_k.strftime('%Y-%m-%d %H:%M')}, TOF {row[1]:.1f}d] -> Total {row[4]*1000:.0f} m/s (Dep {row[2]*1000:.0f}, Cap {row[3]*1000:.0f})")

if __name__ == "__main__":
    run_optimization_scan()
