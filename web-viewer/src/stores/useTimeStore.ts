import { create } from 'zustand';

interface TimeState {
    isPlaying: boolean;
    currentTime: number; // Seconds from epoch
    startTime: number;
    endTime: number;
    timeScale: number; // 1x, 100x, etc.

    setIsPlaying: (playing: boolean) => void;
    setCurrentTime: (time: number) => void;
    setTimeScale: (scale: number) => void;
    setMissionDuration: (start: number, end: number) => void;
    togglePlay: () => void;
}

export const useTimeStore = create<TimeState>((set) => ({
    isPlaying: false,
    currentTime: 0,
    startTime: 0,
    endTime: 100, // Default dummy range
    timeScale: 1, // Real-time default

    setIsPlaying: (playing) => set({ isPlaying: playing }),

    setCurrentTime: (time) => set((state) => ({
        currentTime: Math.max(state.startTime, Math.min(time, state.endTime))
    })),

    setTimeScale: (scale) => set({ timeScale: scale }),

    setMissionDuration: (start, end) => set({
        startTime: start,
        endTime: end,
        currentTime: start
    }),

    togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
}));
