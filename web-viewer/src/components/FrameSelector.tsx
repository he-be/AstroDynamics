import React from 'react';
import { useReferenceFrameStore, type FrameType } from '../stores/useReferenceFrameStore';
import clsx from 'clsx';
import { Globe, RefreshCw, Anchor } from 'lucide-react';

export const FrameSelector: React.FC = () => {
    const { frameType, centerBody, secondaryBody, setFrame } = useReferenceFrameStore();

    const handleSetFrame = (type: FrameType, center: string, secondary?: string) => {
        setFrame(type, center, secondary);
    };

    return (
        <div className="absolute top-4 right-4 flex flex-col gap-2 items-end pointer-events-auto">
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

                {/* Rotating (Synodic) */}
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

                <button
                    onClick={() => handleSetFrame('rotating', 'jupiter', 'europa')}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors text-left",
                        (frameType === 'rotating' && secondaryBody === 'europa') ? "bg-purple-900/50 text-purple-300 border border-purple-700/50" : "hover:bg-white/10 text-gray-300"
                    )}
                >
                    <RefreshCw size={14} />
                    <span>Jupiter-Europa</span>
                </button>

                {/* Body Fixed (Centered) */}
                <div className="h-px bg-white/10 my-1" />
                <div className="text-[10px] font-bold text-gray-500 px-1">CENTER ON</div>

                {['ganymede', 'europa', 'io'].map(moon => (
                    <button
                        key={moon}
                        onClick={() => handleSetFrame('body-fixed', moon)}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1 rounded text-xs transition-colors",
                            (frameType === 'body-fixed' && centerBody === moon) ? "bg-amber-900/30 text-amber-300" : "hover:bg-white/10 text-gray-400"
                        )}
                    >
                        <Anchor size={12} />
                        <span className="capitalize">{moon}</span>
                    </button>
                ))}

            </div>
        </div>
    );
};
