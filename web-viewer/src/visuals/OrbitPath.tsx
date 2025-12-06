import React, { useMemo } from 'react';
import { useMissionStore } from '../stores/useMissionStore';
import { useTimeStore } from '../stores/useTimeStore';
import { useEphemeris } from '../hooks/useEphemeris';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import { Vector3, CatmullRomCurve3 } from 'three';
import { Line } from '@react-three/drei';

interface OrbitPathProps {
    bodyName: string;
    color: string;
}

// Orbital Periods in seconds
const PERIODS: Record<string, number> = {
    'io': 1.769 * 86400,
    'europa': 3.551 * 86400,
    'ganymede': 7.154 * 86400,
    'callisto': 16.689 * 86400,
};

// Sampling Step logic to avoid too many points?
// Timeline resolution is usually adequate.

export const OrbitPath: React.FC<OrbitPathProps> = ({ bodyName, color }) => {
    const showOrbits = useVisualizationStore(s => s.showOrbits);
    const manifest = useMissionStore(s => s.manifest);
    const currentTime = useTimeStore(s => s.currentTime);
    const { transform } = useEphemeris();

    const points = useMemo(() => {
        if (!showOrbits || !manifest) return null;

        const period = PERIODS[bodyName.toLowerCase()];
        if (!period) return null;

        // Window: [Today - Period, Today + Period]
        const tStart = currentTime - period;
        const tEnd = currentTime + period;

        const pts: Vector3[] = [];
        const timeline = manifest.timeline;

        // Sampling: If timeline is dense, skip points?
        // Let's take every Nth point to keep segment usage low.
        // Assuming timeline is 100-3000 points.
        const step = 1;

        for (let i = 0; i < timeline.length; i += step) {
            const entry = timeline[i];

            // Filter by Window
            // We want to draw the orbit available in data.
            // If data starts at t=0 and currentTime=0, we can only draw Future.
            // If currentTime=End, we can only draw Past.
            if (entry.time >= tStart && entry.time <= tEnd) {
                // Extract Body Position
                // entry.bodies might be undefined in some old formats, but we updated it.
                if (entry.bodies && entry.bodies[bodyName.toLowerCase()]) {
                    const raw = entry.bodies[bodyName.toLowerCase()];
                    const v = new Vector3(raw[0], raw[1], raw[2]);

                    // Apply Transform (Inertial -> Current Frame)
                    v.add(transform.offset);
                    v.applyQuaternion(transform.rotation);

                    pts.push(v);
                }
            }
        }

        if (pts.length < 2) return null;

        // Spline Interpolation for smoothness
        const curve = new CatmullRomCurve3(pts);
        // Density: ~1 point per segment is jagged. We want ~20.
        return curve.getPoints(pts.length * 20);
    }, [showOrbits, manifest, currentTime, bodyName, transform]);

    if (!points) return null;

    return (
        <Line
            points={points}
            color={color}
            opacity={0.3}
            transparent
            lineWidth={1}
            dashed
            dashScale={50} // Adjust scale for visibility
            gapSize={20}
        />
    );
};
