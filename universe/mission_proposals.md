# Mission Scenarios Proposal (Golden Master Tests)

To verify the AI Agent's capabilities via MCP, we will first implement these "Ground Truth" scenarios in Python. The Agent's success will be measured by how closely it can match or exceed the performance (Fuel Efficiency, Precision, Time) of these reference scripts.

## Philosophy: "The Python Golden Master"
Each scenario will be implemented as a standalone Python script (e.g., `scenarios/mission_gravity_assist.py`).
- **Input**: Mission Goals (Start, End, Constraints).
- **Execution**: Best-effort algorithmic solution (using `optimization.py`, `mission.py`).
- **Output**: Telemetry Log + Performance Metrics (Delta V, Duration).

The Agent will be given the *Input* and must produce the *Output* via MCP tools.

---

## Proposed Scenarios

### 1. The Europa Slingshot (Efficiency Test)
**Objective**: Fly from Ganymede to Io, using a **Gravity Assist** at Europa to reduce Delta V.
- **Context**: Direct transfer Ganymede->Io is expensive (~3 km/s). Passing behind Europa can brake the spacecraft relative to Jupiter.
- **Complexity**: High. Requires timing the launch to align Ganymede, Europa, and Io (Synodic phasing).
- **Python Implementation**:
    - Extension of `PorkchopOptimizer` to support "3-Body Flyby" search (Start->Flyby->End).
    - Lambert Solver for Leg 1 (Gan->Eur) and Leg 2 (Eur->Io).
    - Matching V_inf at Europa (Flyby constraint).
- **Metric**: Total Delta V < 2.5 km/s (Benchmark vs Direct).

### 2. The Lagrangian Sentinel (Control Test)
**Objective**: Maintain position at the unstable **Ganymede-Jupiter L2 point** for 30 days.
- **Context**: L2 is dynamically unstable. Without correction, the ship will drift away exponentially.
- **Complexity**: Medium (Control Theory). Requires periodic "Station Keeping" maneuvers.
- **Python Implementation**:
    - `FlightController` monitor drift.
    - If drift > Threshold (e.g., 10km), execute slight maneuver to return to manifold.
- **Metric**: Fuel consumption per day (Station Keeping Cost). Agent must minimize this.

### 3. The Inclined Rescue (3D Geometry Test)
**Objective**: Intercept a "distressed ship" in a **High Inclination (45°)** orbit around Callisto.
- **Context**: Our moons are planar. A polar/inclined orbit requires a costly "Plane Change".
- **Complexity**: 3D Vector Math. Plane change is cheapest at apoapsis/nodes.
- **Python Implementation**:
    - Target state defined in 3D (Inclined).
    - Optimizer must find the node crossing or optimal single-burn solution.
- **Metric**: Miss Distance < 10 km. Time to Intercept.

### 4. The Sample Return (Multi-Stage Test)
**Objective**: **Land** on Europa, collect sample, and **Return** to Ganymede.
- **Context**: Full mission cycle. Deorbit, (Simulation of landing simplified as reaching surface velocity 0?), Ascent, Return.
- **Complexity**: Very High.
- **Python Implementation**:
    - Simplified "Touch and Go": Match Surface Velocity -> Wait -> Burn for Ascent -> Return Transfer.
- **Metric**: Mission Success (All phases complete).

---

## Recommended Dev Plan (TDD)
1. **Implement Scenario 2 (Sentinel)**:
   - Why: Adds "Continuous Control" logic to our `FlightController`, which is essential for an Agent.
   - New Tech: `StationKeeping` logic.
2. **Implement Scenario 1 (Slingshot)**:
   - Why: Adds "Multi-Leg Optimization" to `Porkchop`. Crucial for "Agent Planning" intelligence.
   - New Tech: `FlybyPlanner`.
3. **Implement Scenario 3 (Rescue)**:
   - Why: Stress tests 3D Lambert solver (checking for singularities or plane errors).

Please review and select which scenarios to prioritize.
