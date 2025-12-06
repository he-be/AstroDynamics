import { describe, it, expect, beforeEach } from 'vitest';
import { useReferenceFrameStore } from './useReferenceFrameStore';
import { act } from '@testing-library/react';

describe('useReferenceFrameStore', () => {
    beforeEach(() => {
        useReferenceFrameStore.setState({ frameType: 'inertial', centerBody: 'jupiter', secondaryBody: undefined });
    });

    it('updates frame state', () => {
        expect(useReferenceFrameStore.getState().frameType).toBe('inertial');

        act(() => {
            useReferenceFrameStore.getState().setFrame('rotating', 'jupiter', 'ganymede');
        });

        expect(useReferenceFrameStore.getState().frameType).toBe('rotating');
        expect(useReferenceFrameStore.getState().centerBody).toBe('jupiter');
        expect(useReferenceFrameStore.getState().secondaryBody).toBe('ganymede');
    });
});
