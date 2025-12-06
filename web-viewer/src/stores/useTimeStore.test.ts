import { describe, it, expect, beforeEach } from 'vitest';
import { useTimeStore } from './useTimeStore';
import { act } from '@testing-library/react';

describe('useTimeStore', () => {
    beforeEach(() => {
        useTimeStore.setState({ isPlaying: false, currentTime: 0, timeScale: 1 });
    });

    it('toggles play state', () => {
        expect(useTimeStore.getState().isPlaying).toBe(false);
        act(() => {
            useTimeStore.getState().togglePlay();
        });
        expect(useTimeStore.getState().isPlaying).toBe(true);
    });

    it('sets time scale', () => {
        act(() => useTimeStore.getState().setTimeScale(100));
        expect(useTimeStore.getState().timeScale).toBe(100);
    });

    it('clamps current time within bounds', () => {
        useTimeStore.setState({ startTime: 0, endTime: 100 });
        act(() => useTimeStore.getState().setCurrentTime(150));
        expect(useTimeStore.getState().currentTime).toBe(100);

        act(() => useTimeStore.getState().setCurrentTime(-50));
        expect(useTimeStore.getState().currentTime).toBe(0);
    });
});
