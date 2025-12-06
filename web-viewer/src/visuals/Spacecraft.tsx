import React, { useMemo } from 'react';
import { useMissionStore } from '../stores/useMissionStore';
import { useTimeStore } from '../stores/useTimeStore';
import { useEphemeris } from '../hooks/useEphemeris';
import { Vector3 } from 'three';
import { Line, Html } from '@react-three/drei';
import { useVisualizationStore } from '../stores/useVisualizationStore';

const UNIT_SCALE = 0.001;

export const Spacecraft: React.FC = () => {
    const currentTime = useTimeStore(s => s.currentTime);
    const getInterpolatedState = useMissionStore(s => s.getInterpolatedState);
    const manifest = useMissionStore(s => s.manifest);
    const { transform } = useEphemeris();
    const showOutlines = useVisualizationStore(s => s.showOutlines);

    // 1. Get Current State (Interpolated)
    const state = getInterpolatedState(currentTime);

    // 2. Render Position (Transformed)
    const renderPos = useMemo(() => {
        if (!state) return null;
        const pos = new Vector3(state.position[0], state.position[1], state.position[2]);
        pos.add(transform.offset);
        pos.applyQuaternion(transform.rotation);
        return pos;
    }, [state, transform]);

    // 3. Trajectory Trail (Transformed)
    // We compute points for the whole trail? Or just window?
    // Computing whole trail every frame is heavy.
    // Optimization: Just show trail in Inertial Frame? No, users want to see path in Rotating Frame.
    // We must transform the points. Limiting sample count is key.
    const trailPoints = useMemo(() => {
        if (!manifest) return [];

        // Downsample for performance (e.g. every 10th point)
        const points = [];
        const step = Math.max(1, Math.floor(manifest.timeline.length / 500)); // Max 500 points

        for (let i = 0; i < manifest.timeline.length; i += step) {
            const p = manifest.timeline[i];

            // Stop drawing if point is in the future
            if (p.time > currentTime) break;

            // WARNING: Trail transformation relies on the Current Time's transform.
            // If we are in Rotating Frame, the "Trail" changes shape as we play!
            // That is correct for an "Instantaneous Path" relative to the frame.
            // But usually we want the path AS FLOWN in that frame.

            // Wait. If I just apply (offset, rotation) of CURRENT time to OLD points, 
            // that means I am rotating the OLD inertial points by CURRENT angle.
            // That effectively shows "Where those inertial points are relative to me NOW".
            // This produces the correct "Snail Trail" in a Rotating Frame.

            const v = new Vector3(p.position[0], p.position[1], p.position[2]);
            v.add(transform.offset);
            v.applyQuaternion(transform.rotation);
            points.push(v);
        }
        return points;
    }, [manifest, transform]); // Recalculates when transform changes (every frame if rotating!)

    if (!state || !renderPos) return null;

    // Hide Spacecraft if it's just the default Ephemeris
    if (manifest?.meta.missionName === 'Solar System Ephemeris') return null;



    return (
        <group scale={[UNIT_SCALE, UNIT_SCALE, UNIT_SCALE]}>
            <mesh position={renderPos}>
                <coneGeometry args={[500, 1000, 8]} />
                <meshStandardMaterial color="cyan" emissive="cyan" emissiveIntensity={0.8} />
            </mesh>

            {trailPoints.length > 1 && (
                <Line
                    points={trailPoints}
                    color="cyan"
                    opacity={0.5}
                    transparent
                    lineWidth={1}
                />
            )}

            {showOutlines && (
                <Html position={renderPos} zIndexRange={[60, 0]}>
                    <div className="relative pointer-events-none">
                        {/* Dot: Centered on [0,0,0] */}
                        <div className="absolute top-0 left-0 w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,1)]"
                            style={{ transform: 'translate(-50%, -50%)' }} />

                        {/* Label: Centered horizontally, pushed down */}
                        <div className="absolute top-3 left-0 text-[10px] text-cyan-300 font-mono whitespace-nowrap bg-black/60 px-1 rounded backdrop-blur-sm border border-cyan-500/30"
                            style={{ transform: 'translate(-50%, 0)' }}>
                            Spacecraft
                        </div>
                    </div>
                </Html>
            )}
        </group>
    );
};
