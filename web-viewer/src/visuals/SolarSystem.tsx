import React, { useMemo } from 'react';
import { Body } from './Body';
import { useEphemeris } from '../hooks/useEphemeris';
import { useReferenceFrameStore } from '../stores/useReferenceFrameStore';

const UNIT_SCALE = 0.001;

export const SolarSystem: React.FC = () => {
    const { frameType } = useReferenceFrameStore();
    const { bodies, transform } = useEphemeris();

    // Apply Transform to all bodies
    const renderBodies = useMemo(() => {
        return Object.values(bodies).map(b => {
            // 1. Translate
            const pos = b.position.clone().add(transform.offset);
            // 2. Rotate
            pos.applyQuaternion(transform.rotation);

            return { ...b, renderPos: pos };
        });
    }, [bodies, transform]);

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
