import React, { useMemo } from 'react';
import { useReferenceFrameStore, type FrameType } from '../stores/useReferenceFrameStore';
import { useMissionStore } from '../stores/useMissionStore';
import clsx from 'clsx';
import { Globe, RefreshCw, Anchor, Rocket } from 'lucide-react';


export const FrameSelector: React.FC = () => {
    const { frameType, centerBody, secondaryBody, setFrame } = useReferenceFrameStore();
    const manifest = useMissionStore(s => s.manifest);

    const availableBodies = useMemo(() => {
        const set = new Set<string>();
        set.add('spacecraft'); // Always available
        set.add('jupiter');    // Always available

        if (manifest?.meta?.bodies) {
            manifest.meta.bodies.forEach(b => set.add(b.toLowerCase()));
        } else {
            // Fallbacks if no manifest loaded yet
            ['io', 'europa', 'ganymede', 'callisto'].forEach(b => set.add(b));
        }
        return Array.from(set);
    }, [manifest]);

    const handleSetFrame = (type: FrameType, center: string, secondary?: string) => {
        setFrame(type, center, secondary);
    };

    return (
        <div className="fixed z-[100] top-4 right-4 flex flex-col gap-2 items-end pointer-events-auto">
            <div className="bg-[#1a1a24]/90 backdrop-blur-md border border-white/10 rounded-lg p-2 text-white shadow-xl flex flex-col gap-2 min-w-[200px]">
                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1 px-1">Reference Frame</div>

                {/* Inertial */}
                <button
                    onClick={() => handleSetFrame('inertial', 'jupiter')}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors text-left",
                        frameType === 'inertial' ? "bg-cyan-900/50 text-cyan-300 border border-cyan-700/50" : "hover:bg-white/10 text-gray-300"
                    )}
                >
                    <Globe size={14} />
                    <span>Jovicentric Inertial</span>
                </button>

                {/* Rotating (Synodic) - Only if Ganymede/Europa available */}
                <button
                    onClick={() => handleSetFrame('rotating', 'jupiter', 'ganymede')}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors text-left",
                        (frameType === 'rotating' && secondaryBody === 'ganymede') ? "bg-purple-900/50 text-purple-300 border border-purple-700/50" : "hover:bg-white/10 text-gray-300"
                    )}
                >
                    <RefreshCw size={14} />
                    <span>Jupiter-Ganymede (L2)</span>
                </button>

                {/* Body Fixed (Centered) */}
                <div className="h-px bg-white/10 my-1" />
                <div className="text-[10px] font-bold text-gray-500 px-1">FOCUS OBJECT</div>

                <div className="flex flex-col gap-1 max-h-[300px] overflow-y-auto custom-scrollbar">
                    {availableBodies.map(body => (
                        <button
                            key={body}
                            onClick={() => handleSetFrame('body-fixed', body)}
                            className={clsx(
                                "flex items-center gap-2 px-3 py-1 rounded text-xs transition-colors capitalize",
                                (frameType === 'body-fixed' && centerBody === body) ? "bg-amber-900/30 text-amber-300 border border-amber-700/30" : "hover:bg-white/10 text-gray-400"
                            )}
                        >
                            {body === 'spacecraft' ? <Rocket size={12} /> : <Anchor size={12} />}
                            <span>{body}</span>
                        </button>
                    ))}
                </div>

            </div>
        </div>
    );
};
