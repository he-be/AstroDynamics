import { create } from 'zustand';

export type FrameType = 'inertial' | 'body-fixed' | 'rotating';

interface ReferenceFrameState {
    frameType: FrameType;
    centerBody: string; // 'jupiter', 'ganymede', etc.
    secondaryBody?: string; // For rotating frame (e.g. 'ganymede' in Jupiter-Ganymede frame)

    setFrame: (type: FrameType, center: string, secondary?: string) => void;
}

export const useReferenceFrameStore = create<ReferenceFrameState>((set) => ({
    frameType: 'inertial',
    centerBody: 'jupiter',
    secondaryBody: undefined,

    setFrame: (type, center, secondary) => set({
        frameType: type,
        centerBody: center,
        secondaryBody: secondary
    }),
}));
