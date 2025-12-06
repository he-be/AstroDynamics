import { create } from 'zustand';

interface VisualizationState {
    showOutlines: boolean;
    showOrbits: boolean;
    showLabels: boolean;

    toggleOutlines: () => void;
    toggleOrbits: () => void;
    toggleLabels: () => void;
}

export const useVisualizationStore = create<VisualizationState>((set) => ({
    showOutlines: false,
    showOrbits: true, // Default to true for orbits as they are helpful
    showLabels: true,

    toggleOutlines: () => set((state) => ({ showOutlines: !state.showOutlines })),
    toggleOrbits: () => set((state) => ({ showOrbits: !state.showOrbits })),
    toggleLabels: () => set((state) => ({ showLabels: !state.showLabels })),
}));
