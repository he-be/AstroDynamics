
import React, { useMemo } from 'react';
import { useMissionStore } from '../stores/useMissionStore';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import { useEphemeris } from '../hooks/useEphemeris';
import { Vector3 } from 'three';
import { Html } from '@react-three/drei';

const UNIT_SCALE = 0.001; // Scaler for visibility

export const ManeuverMarkers: React.FC = () => {
    const manifest = useMissionStore(s => s.manifest);
    const { selectedManeuverIndex } = useVisualizationStore();
    const { transform } = useEphemeris();

    // Pre-calculate marker positions and data
    const markers = useMemo(() => {
        if (!manifest || !manifest.maneuvers) return [];

        return manifest.maneuvers.map((m, index) => {
            // 1. Find approximate position from timeline
            // Accessing store's getInterpolatedState inside a loop might be okay if memosized properly, 
            // but we don't have access to the hook's returned function here easily without context.
            // We can just implement simple search/interp here or blindly use nearest point.
            // Timeline is sorted.

            // Binary search or Find
            // Since we map ALL maneuvers, and they are ordered? 
            // Maneuvers might NOT be ordered in manifest? (Usually yes).

            // Simple approach: Find closest timeline point
            // Optimization: Assume timeline is dense enough.

            // Let's implement a simple binary search for closest point
            let lo = 0, hi = manifest.timeline.length - 1;
            while (lo <= hi) {
                const mid = (lo + hi) >> 1;
                if (manifest.timeline[mid].time < m.startTime) lo = mid + 1;
                else hi = mid - 1;
            }
            // index 'lo' is first element >= m.startTime
            const idx = Math.min(Math.max(lo, 0), manifest.timeline.length - 1);
            const state = manifest.timeline[idx];
            // Note: This is nearest neighbor, not valid interpolation. 
            // For visualization "markers", likely acceptable. 
            // Improving to Linear Interp is better if easy.

            const p = state.position;
            const pos = new Vector3(p[0], p[1], p[2]);

            // Apply Transform (Inertial -> Current Frame)
            pos.add(transform.offset);
            pos.applyQuaternion(transform.rotation);

            // Delta V Vector
            const dv = new Vector3(m.deltaV[0], m.deltaV[1], m.deltaV[2]);
            const dvMag = dv.length(); // km/s

            // Transform Vector Direction (Rotate only)
            const dvDir = dv.clone().normalize();
            dvDir.applyQuaternion(transform.rotation);

            return {
                index,
                position: pos,
                direction: dvDir,
                magnitude: dvMag,
                data: m
            };
        });
    }, [manifest, transform]);

    if (!manifest || !manifest.maneuvers) return null;

    return (
        <group scale={[UNIT_SCALE, UNIT_SCALE, UNIT_SCALE]}>
            {markers.map((marker) => {
                const isSelected = selectedManeuverIndex === marker.index;
                const color = isSelected ? "#00ffff" : "#ffaa00";
                const length = isSelected ? 4000 : 2000; // Fixed visual size

                return (
                    <group
                        key={marker.index}
                        position={marker.position}
                    // Quaternions for arrow orientation?
                    // arrowHelper handles direction. 
                    // But if we use mesh, we need lookAt.
                    >
                        {/* Using arrowHelper primitive */}
                        <arrowHelper
                            args={[marker.direction, new Vector3(0, 0, 0), length, color, length * 0.2, length * 0.2]}
                        />

                        {isSelected && (
                            <Html position={[0, length, 0]} zIndexRange={[100, 0]}>
                                <div className="bg-black/80 text-cyan-400 border border-cyan-500/50 px-2 py-1 rounded text-xs font-mono whitespace-nowrap backdrop-blur-md transform -translate-x-1/2 -translate-y-full">
                                    <div>ΔV: {(marker.magnitude * 1000).toFixed(1)} m/s</div>
                                    <div>T+: {(marker.data.startTime / 86400).toFixed(1)}d</div>
                                </div>
                            </Html>
                        )}
                    </group>
                );
            })}
        </group>
    );
};
