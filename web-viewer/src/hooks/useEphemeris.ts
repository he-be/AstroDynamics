import { useMemo } from 'react';
import { Vector3, Quaternion, Euler } from 'three';
import { useTimeStore } from '../stores/useTimeStore';
import { useReferenceFrameStore } from '../stores/useReferenceFrameStore';
import { useMissionStore } from '../stores/useMissionStore';

// Constants (km)
const R_JUPITER = 71492;
const R_IO = 1821;
const R_EUROPA = 1560;
const R_GANYMEDE = 2634;
const R_CALLISTO = 2410;

// Mock Orbital Elements removed.


export interface BodyState {
    name: string;
    radius: number;
    color: string;
    position: Vector3;
    scaleFactor: number;
}

export function useEphemeris() {
    const currentTime = useTimeStore((state) => state.currentTime);
    const { frameType, centerBody, secondaryBody } = useReferenceFrameStore();

    // 1. Calculate Positions
    const getInterpolatedState = useMissionStore((state) => state.getInterpolatedState);
    const missionState = getInterpolatedState(currentTime);

    const inertialBodies = useMemo(() => {
        const bodies: Record<string, BodyState> = {};

        // Base Definitions (Initialized at Origin, waiting for Data)
        bodies['jupiter'] = { name: 'Jupiter', radius: R_JUPITER, color: '#d4a373', position: new Vector3(0, 0, 0), scaleFactor: 1 };
        bodies['io'] = { name: 'Io', radius: R_IO, color: '#f4e409', position: new Vector3(), scaleFactor: 5 };
        bodies['europa'] = { name: 'Europa', radius: R_EUROPA, color: '#a67c52', position: new Vector3(), scaleFactor: 5 };
        bodies['ganymede'] = { name: 'Ganymede', radius: R_GANYMEDE, color: '#706f6f', position: new Vector3(), scaleFactor: 5 };
        bodies['callisto'] = { name: 'Callisto', radius: R_CALLISTO, color: '#5c5c5c', position: new Vector3(), scaleFactor: 5 };

        if (missionState && missionState.bodies) {
            // Apply Loaded Data
            for (const [name, rawPos] of Object.entries(missionState.bodies)) {
                const lowerName = name.toLowerCase();
                if (bodies[lowerName]) {
                    const pos = rawPos as [number, number, number];
                    bodies[lowerName].position.set(pos[0], pos[1], pos[2]);
                }
            }
        }

        // Add Spacecraft (for centering purposes)
        if (missionState) {
            bodies['spacecraft'] = {
                name: 'Spacecraft',
                radius: 0.001, // Tiny radius to avoid visual clutter if rendered
                color: '#00ffff',
                position: new Vector3(missionState.position[0], missionState.position[1], missionState.position[2]),
                scaleFactor: 1
            };
        } else {
            // Fallback if no mission loaded yet, or just ignore
            bodies['spacecraft'] = {
                name: 'Spacecraft',
                radius: 0.001,
                color: '#00ffff',
                position: new Vector3(0, 0, 0),
                scaleFactor: 1
            };
        }

        return bodies;
    }, [currentTime, missionState]);

    // 2. Calculate Frame Transform
    const transform = useMemo(() => {
        let offset = new Vector3(0, 0, 0);
        let rotation = new Quaternion();

        // J2000 to Jovicentric Equatorial Correction
        // Aligns Jupiter's North Pole (ICRF) with Scene Y-axis (Three.js Up)
        const alpha = 268.0565 * (Math.PI / 180);
        const delta = 64.4953 * (Math.PI / 180);
        const pole = new Vector3(
            Math.cos(delta) * Math.cos(alpha),
            Math.cos(delta) * Math.sin(alpha),
            Math.sin(delta)
        );
        const up = new Vector3(0, 1, 0);
        const qTilt = new Quaternion().setFromUnitVectors(pole, up);

        if (centerBody && inertialBodies[centerBody]) {
            offset = inertialBodies[centerBody].position.clone().negate();
        }

        if (frameType === 'inertial') {
            // Apply TILT correction to make orbital plane horizontal
            rotation.copy(qTilt);
        } else if (frameType === 'rotating' && secondaryBody && inertialBodies[secondaryBody]) {
            const jupPos = inertialBodies['jupiter'].position;
            const secPos = inertialBodies[secondaryBody].position;

            // Vector in ICRF
            const relVec = secPos.clone().sub(jupPos);

            // Rotate to Equatorial Plane to find Phase Angle correctly
            relVec.applyQuaternion(qTilt); // Now in XZ plane (approx)

            const currentAngle = Math.atan2(relVec.z, relVec.x);
            const euler = new Euler(0, -currentAngle, 0);
            const qSynodic = new Quaternion().setFromEuler(euler);

            // Total = Synodic * Tilt
            rotation.multiplyQuaternions(qSynodic, qTilt);
        } else {
            // Fallback for centered body-fixed without rotation?
            // Usually body-fixed is just centered inertial?
            // If 'body-fixed', we typically just center.
            // But let's also Apply Tilt so it's consistent?
            // Current code 'body-fixed' sets offset but rotation identity.
            // Let's also tilt it so it's not confusing.
            if (frameType === 'body-fixed') {
                rotation.copy(qTilt);
            }
        }

        return { offset, rotation };
    }, [inertialBodies, frameType, centerBody, secondaryBody]);

    return { bodies: inertialBodies, transform };
}
