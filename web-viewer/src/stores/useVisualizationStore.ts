import { create } from 'zustand';

interface VisualizationState {
    showOutlines: boolean;
    showOrbits: boolean;
    showLabels: boolean;
    selectedManeuverIndex: number | null;

    toggleOutlines: () => void;
    toggleOrbits: () => void;
    toggleLabels: () => void;
    setSelectedManeuver: (index: number | null) => void;
}

export const useVisualizationStore = create<VisualizationState>((set) => ({
    showOutlines: false,
    showOrbits: true, // Default to true for orbits as they are helpful
    showLabels: true,
    selectedManeuverIndex: null,

    toggleOutlines: () => set((state) => ({ showOutlines: !state.showOutlines })),
    toggleOrbits: () => set((state) => ({ showOrbits: !state.showOrbits })),
    toggleLabels: () => set((state) => ({ showLabels: !state.showLabels })),
    setSelectedManeuver: (index) => set({ selectedManeuverIndex: index }),
}));
