import React from 'react';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import clsx from 'clsx';
import { Activity, Eye } from 'lucide-react';

export const ViewSettings: React.FC = () => {
    const { showOutlines, showOrbits, toggleOutlines, toggleOrbits } = useVisualizationStore();

    return (
        <div className="flex flex-col gap-2 pointer-events-auto">
            <div className="bg-[#1a1a24]/90 backdrop-blur-md border border-white/10 rounded-lg p-2 text-white shadow-xl flex flex-col gap-1 min-w-[140px]">
                <div className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1 px-1">Visuals</div>

                <button
                    onClick={toggleOutlines}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors",
                        showOutlines ? "bg-blue-900/50 text-blue-300 border border-blue-700/50" : "hover:bg-white/10 text-gray-400"
                    )}
                >
                    <Eye size={12} />
                    <span>Highlights</span>
                </button>

                <button
                    onClick={toggleOrbits}
                    className={clsx(
                        "flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors",
                        showOrbits ? "bg-blue-900/50 text-blue-300 border border-blue-700/50" : "hover:bg-white/10 text-gray-400"
                    )}
                >
                    <Activity size={12} />
                    <span>Orbits</span>
                </button>
            </div>
        </div>
    );
};
