import { describe, it, expect } from 'vitest';
import { useMissionStore } from '../useMissionStore';
import type { MissionManifest } from '../../types/mission';

describe('useMissionStore', () => {
    it('interpolate bodies correctly', () => {
        const store = useMissionStore.getState();

        const manifest: MissionManifest = {
            meta: {
                missionName: 'Test',
                startTime: '2025-01-01T00:00:00Z',
                endTime: '2025-01-02T00:00:00Z',
                bodies: ['jupiter', 'ganymede']
            },
            timeline: [
                {
                    time: 0,
                    position: [0, 0, 0],
                    velocity: [0, 0, 0],
                    mass: 1000,
                    bodies: {
                        'ganymede': [100, 0, 0]
                    }
                },
                {
                    time: 10,
                    position: [100, 0, 0],
                    velocity: [0, 0, 0],
                    mass: 1000,
                    bodies: {
                        'ganymede': [200, 0, 0]
                    }
                }
            ],
            maneuvers: []
        };

        store.loadManifest(manifest);

        // Test at t=5 (Midpoint)
        const state = store.getInterpolatedState(5);

        expect(state).not.toBeNull();
        expect(state?.bodies).toBeDefined();
        expect(state?.bodies?.ganymede).toBeDefined();

        const ganymede = state!.bodies!.ganymede;
        expect(ganymede[0]).toBeCloseTo(150); // Linear interpolation: 100 + (200-100)*0.5
        expect(ganymede[1]).toBeCloseTo(0);
        expect(ganymede[2]).toBeCloseTo(0);
    });
});
