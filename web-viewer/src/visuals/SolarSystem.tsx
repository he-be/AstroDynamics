import React, { useMemo } from 'react';
import { Body } from './Body';
import { useTimeStore } from '../stores/useTimeStore';
import { useReferenceFrameStore } from '../stores/useReferenceFrameStore';
import { Vector3, Quaternion, Euler } from 'three';

// Constants (km)
const R_JUPITER = 71492;
const R_IO = 1821;
const R_EUROPA = 1560;
const R_GANYMEDE = 2634;
const R_CALLISTO = 2410;
const UNIT_SCALE = 0.001;

// Mock Orbital Elements (Approximate)
// Periods in seconds
const T_IO = 1.769 * 86400;
const T_EUROPA = 3.551 * 86400;
const T_GANYMEDE = 7.154 * 86400;
const T_CALLISTO = 16.689 * 86400;

// SMA (km)
const A_IO = 421700;
const A_EUROPA = 671100;
const A_GANYMEDE = 1070400;
const A_CALLISTO = 1882700;

interface BodyState {
    name: string;
    radius: number;
    color: string;
    position: Vector3; // Scaled Units? No, keep in km, scale at render.
    scaleFactor: number;
}

export const SolarSystem: React.FC = () => {
    const currentTime = useTimeStore((state) => state.currentTime);
    const { frameType, centerBody, secondaryBody } = useReferenceFrameStore();

    // 1. Calculate Inertial Positions (Jovicentric)
    const inertialBodies = useMemo(() => {
        const bodies: Record<string, BodyState> = {};

        // Jupiter
        bodies['jupiter'] = { name: 'Jupiter', radius: R_JUPITER, color: '#d4a373', position: new Vector3(0, 0, 0), scaleFactor: 1 };

        // Helper for circular orbit
        const getPos = (a: number, T: number, phase: number = 0) => {
            // n = 2pi / T
            const n = (2 * Math.PI) / T;
            const angle = n * currentTime + phase;
            return new Vector3(a * Math.cos(angle), 0, a * Math.sin(angle)); // X-Z plane orbit
        };

        bodies['io'] = { name: 'Io', radius: R_IO, color: '#f4e409', position: getPos(A_IO, T_IO, 0), scaleFactor: 5 };
        bodies['europa'] = { name: 'Europa', radius: R_EUROPA, color: '#a67c52', position: getPos(A_EUROPA, T_EUROPA, 1.0), scaleFactor: 5 };
        bodies['ganymede'] = { name: 'Ganymede', radius: R_GANYMEDE, color: '#706f6f', position: getPos(A_GANYMEDE, T_GANYMEDE, 2.0), scaleFactor: 5 };
        bodies['callisto'] = { name: 'Callisto', radius: R_CALLISTO, color: '#5c5c5c', position: getPos(A_CALLISTO, T_CALLISTO, 3.0), scaleFactor: 5 };

        return bodies;
    }, [currentTime]);

    // 2. Determine Transform (Offset & Rotation)
    const transform = useMemo(() => {
        let offset = new Vector3(0, 0, 0);
        let rotation = new Quaternion();

        // Center Body Offset
        if (centerBody && inertialBodies[centerBody]) {
            offset = inertialBodies[centerBody].position.clone().negate();
        }

        // Rotating Frame (Rotation)
        if (frameType === 'rotating' && secondaryBody && inertialBodies[secondaryBody]) {
            // We want the primary (Jupiter) -> Secondary vector to be fixed on +X axis.
            // Assuming Center is Jupiter.
            // If center is not Jupiter, this logic gets tricky, but we usually rotate about the primary.

            // Let's assume we rotate around the current 'centerBody' (which should be Jupiter for normal L-points).
            // Or strictly rotate around Jupiter.

            // Vector from Jupiter to Secondary (Inertial)
            const jupPos = inertialBodies['jupiter'].position;
            const secPos = inertialBodies[secondaryBody].position;
            const relVec = secPos.clone().sub(jupPos);

            // Angle in X-Z plane
            const currentAngle = Math.atan2(relVec.z, relVec.x);

            // We want to rotate by -currentAngle so relVec becomes (Mag, 0, 0)
            const euler = new Euler(0, -currentAngle, 0);
            rotation.setFromEuler(euler);
        }

        return { offset, rotation };
    }, [inertialBodies, frameType, centerBody, secondaryBody]);

    // 3. Apply Transform to all bodies
    const renderBodies = useMemo(() => {
        return Object.values(inertialBodies).map(b => {
            // 1. Translate (so center is at 0,0,0)
            const pos = b.position.clone().add(transform.offset);

            // 2. Rotate (if needed) - Rotate around 0,0,0
            pos.applyQuaternion(transform.rotation);

            return { ...b, renderPos: pos };
        });
    }, [inertialBodies, transform]);


    return (
        <group scale={[UNIT_SCALE, UNIT_SCALE, UNIT_SCALE]}>
            {renderBodies.map(b => (
                <Body
                    key={b.name}
                    name={b.name}
                    radius={b.radius}
                    color={b.color}
                    position={[b.renderPos.x, b.renderPos.y, b.renderPos.z]}
                    scaleFactor={b.scaleFactor}
                />
            ))}

            {/* Visual Guide for Axes if rotating */}
            {frameType === 'rotating' && (
                <gridHelper args={[2000000, 20, 0x444444, 0x222222]} rotation={[0, 0, 0]} />
            )}
        </group>
    );
};
