import { useMemo } from 'react';
import { Vector3, Quaternion, Euler } from 'three';
import { useTimeStore } from '../stores/useTimeStore';
import { useReferenceFrameStore } from '../stores/useReferenceFrameStore';

// Constants (km)
const R_JUPITER = 71492;
const R_IO = 1821;
const R_EUROPA = 1560;
const R_GANYMEDE = 2634;
const R_CALLISTO = 2410;

// Mock Orbital Elements
const T_IO = 1.769 * 86400;
const T_EUROPA = 3.551 * 86400;
const T_GANYMEDE = 7.154 * 86400;
const T_CALLISTO = 16.689 * 86400;

const A_IO = 421700;
const A_EUROPA = 671100;
const A_GANYMEDE = 1070400;
const A_CALLISTO = 1882700;

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

    // 1. Calculate Inertial Positions
    const inertialBodies = useMemo(() => {
        const bodies: Record<string, BodyState> = {};

        bodies['jupiter'] = { name: 'Jupiter', radius: R_JUPITER, color: '#d4a373', position: new Vector3(0, 0, 0), scaleFactor: 1 };

        const getPos = (a: number, T: number, phase: number = 0) => {
            const n = (2 * Math.PI) / T;
            const angle = n * currentTime + phase;
            return new Vector3(a * Math.cos(angle), 0, a * Math.sin(angle));
        };

        bodies['io'] = { name: 'Io', radius: R_IO, color: '#f4e409', position: getPos(A_IO, T_IO, 0), scaleFactor: 5 };
        bodies['europa'] = { name: 'Europa', radius: R_EUROPA, color: '#a67c52', position: getPos(A_EUROPA, T_EUROPA, 1.0), scaleFactor: 5 };
        bodies['ganymede'] = { name: 'Ganymede', radius: R_GANYMEDE, color: '#706f6f', position: getPos(A_GANYMEDE, T_GANYMEDE, 2.0), scaleFactor: 5 };
        bodies['callisto'] = { name: 'Callisto', radius: R_CALLISTO, color: '#5c5c5c', position: getPos(A_CALLISTO, T_CALLISTO, 3.0), scaleFactor: 5 };
        return bodies;
    }, [currentTime]);

    // 2. Calculate Frame Transform
    const transform = useMemo(() => {
        let offset = new Vector3(0, 0, 0);
        let rotation = new Quaternion();

        if (centerBody && inertialBodies[centerBody]) {
            offset = inertialBodies[centerBody].position.clone().negate();
        }

        if (frameType === 'rotating' && secondaryBody && inertialBodies[secondaryBody]) {
            const jupPos = inertialBodies['jupiter'].position;
            const secPos = inertialBodies[secondaryBody].position;
            const relVec = secPos.clone().sub(jupPos);
            const currentAngle = Math.atan2(relVec.z, relVec.x);
            const euler = new Euler(0, -currentAngle, 0);
            rotation.setFromEuler(euler);
        }

        return { offset, rotation };
    }, [inertialBodies, frameType, centerBody, secondaryBody]);

    return { bodies: inertialBodies, transform };
}
