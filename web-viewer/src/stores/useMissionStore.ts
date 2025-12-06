import { create } from 'zustand';
import type { MissionManifest, TrajectoryState } from '../types/mission';

interface MissionStore {
    manifest: MissionManifest | null;
    loadManifest: (data: MissionManifest) => void;
    getInterpolatedState: (time: number) => TrajectoryState | null;
}

export const useMissionStore = create<MissionStore>((set, get) => ({
    manifest: null,

    loadManifest: (data) => set({ manifest: data }),

    getInterpolatedState: (time) => {
        const { manifest } = get();
        if (!manifest || manifest.timeline.length < 2) return null;

        // Binary search for the segment
        const { timeline } = manifest;
        let lo = 0, hi = timeline.length - 1;

        // Clamp
        if (time <= timeline[0].time) return timeline[0];
        if (time >= timeline[timeline.length - 1].time) return timeline[timeline.length - 1];

        while (lo <= hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (timeline[mid].time < time) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // timeline[hi] is <= time, timeline[lo] is > time
        const p0 = timeline[hi];
        const p1 = timeline[lo];

        if (!p0 || !p1) return p0 || p1; // Should not happen given clamps

        const t = (time - p0.time) / (p1.time - p0.time);

        // Linear Interpolation for Position
        const pos: [number, number, number] = [
            p0.position[0] + (p1.position[0] - p0.position[0]) * t,
            p0.position[1] + (p1.position[1] - p0.position[1]) * t,
            p0.position[2] + (p1.position[2] - p0.position[2]) * t,
        ];

        // Linear Interpolation for Velocity
        const vel: [number, number, number] = [
            p0.velocity[0] + (p1.velocity[0] - p0.velocity[0]) * t,
            p0.velocity[1] + (p1.velocity[1] - p0.velocity[1]) * t,
            p0.velocity[2] + (p1.velocity[2] - p0.velocity[2]) * t,
        ];

        // Linear for Mass
        const mass = p0.mass + (p1.mass - p0.mass) * t;

        return {
            time,
            position: pos,
            velocity: vel,
            mass
        };
    },
}));
