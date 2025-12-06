export interface Vector3D {
    x: number;
    y: number;
    z: number;
}

export interface Quaternion {
    x: number;
    y: number;
    z: number;
    w: number;
}

export interface MissionEvent {
    time: number;
    type: 'MANEUVER_START' | 'MANEUVER_END' | 'SOI_CHANGE' | 'PERIAPSIS' | 'APOAPSIS' | 'INFO';
    description: string;
}

export interface TrajectoryState {
    time: number; // Seconds from epoch
    position: [number, number, number]; // [x, y, z] km
    velocity: [number, number, number]; // [vx, vy, vz] km/s
    mass: number; // kg
    bodies?: Record<string, [number, number, number]>; // Positions of celestial bodies
}

export interface Maneuver {
    startTime: number;
    duration: number;
    deltaV: [number, number, number];
    type: 'impulsive' | 'finite';
}

export interface MissionManifest {
    meta: {
        missionName: string;
        startTime: string; // ISO 8601
        endTime: string;
        bodies: string[];
    };
    timeline: TrajectoryState[];
    maneuvers: Maneuver[];
    events?: MissionEvent[];
}
