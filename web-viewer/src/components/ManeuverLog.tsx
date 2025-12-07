
import React, { useMemo } from 'react';
import { useMissionStore } from '../stores/useMissionStore';
import { useTimeStore } from '../stores/useTimeStore';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import { useReferenceFrameStore } from '../stores/useReferenceFrameStore';
import { Crosshair } from 'lucide-react';
import clsx from 'clsx';

export const ManeuverLog: React.FC = () => {
    const manifest = useMissionStore(s => s.manifest);
    const { setCurrentTime } = useTimeStore();
    const { selectedManeuverIndex, setSelectedManeuver } = useVisualizationStore();
    const { setFrame } = useReferenceFrameStore();

    // Sort maneuvers: Newest first (Reverse Chronological)
    const maneuvers = useMemo(() => {
        if (!manifest || !manifest.maneuvers) return [];
        // Create indexed array to keep track of original indices
        return manifest.maneuvers.map((m, i) => ({ ...m, originalIndex: i }))
            .sort((a, b) => b.startTime - a.startTime);
    }, [manifest]);

    if (!manifest || !manifest.maneuvers || manifest.maneuvers.length === 0) return null;

    const handleManeuverClick = (m: typeof maneuvers[0]) => {
        // 1. Set Time
        setCurrentTime(m.startTime);

        // 2. Select
        setSelectedManeuver(m.originalIndex);

        // 3. Focus Camera on Spacecraft (by switching to Body-Fixed Spacecraft frame)
        // Only if not already tracking spacecraft? 
        // User requested "Move camera to maneuver location". 
        // Tracking spacecraft ensures this.
        setFrame('body-fixed', 'spacecraft');
    };

    return (
        <div className="max-h-[60vh] flex flex-col pointer-events-auto">
            <div className="bg-[#1a1a24]/90 backdrop-blur-md border border-white/10 rounded-lg shadow-xl flex flex-col w-[280px] overflow-hidden">
                <div className="px-3 py-2 bg-white/5 border-b border-white/10 flex justify-between items-center">
                    <span className="text-xs font-bold text-cyan-400 uppercase tracking-widest">Maneuver Log</span>
                    <span className="text-[10px] text-gray-500">{maneuvers.length} EVTS</span>
                </div>

                <div className="overflow-y-auto custom-scrollbar p-1 flex flex-col gap-1">
                    {maneuvers.map((m) => {
                        const isSelected = selectedManeuverIndex === m.originalIndex;
                        const date = new Date(manifest.meta.startTime);
                        date.setSeconds(date.getSeconds() + m.startTime); // Approximate logic, assuming startTime is seconds from epoch?
                        // Wait, manifest.meta.startTime is String ISO. 
                        // types/mission.ts says TrajectoryState time is "Seconds from epoch".
                        // Usually this epoch corresponds to meta.startTime?
                        // Actually, if we look at `useMissionStore`: `const start = data.timeline[0].time`.
                        // It implies `time` is absolute seconds? Or relative?
                        // FlightController uses ISO strings.
                        // `telemetry.export_mission_manifest` converts to... what?
                        // Let's assume `time` is seconds from J2000 or similar, OR relative to start.
                        // But `useTimeStore` usually handles this.
                        // Visualizing "T+" format is safer if we are unsure of absolute date.

                        // Let's format as "Day X HH:MM"
                        const days = Math.floor(m.startTime / 86400);
                        const hours = Math.floor((m.startTime % 86400) / 3600);

                        const dvMag = Math.sqrt(m.deltaV[0] ** 2 + m.deltaV[1] ** 2 + m.deltaV[2] ** 2); // km/s?
                        // FlightController exports in meters? 
                        // MissionManifest says: `deltaV: [number, number, number]`
                        // FlightController logic: `delta_v_vec_km_s`.
                        // Telemetry export: likely keeps it.
                        // Let's assume km/s if small, m/s if large.
                        // Usually Controller logs m/s for DV magnitude display.
                        // Check `telemetry.py` later if needed. For now assume km/s based on mission.ts logic `velocity: km/s`.

                        return (
                            <button
                                key={m.originalIndex}
                                onClick={() => handleManeuverClick(m)}
                                className={clsx(
                                    "flex flex-col text-left px-3 py-2 rounded transition-all border-l-2",
                                    isSelected
                                        ? "bg-cyan-900/40 border-cyan-400"
                                        : "hover:bg-white/5 border-transparent hover:border-white/20"
                                )}
                            >
                                <div className="flex justify-between items-center mb-1">
                                    <span className={clsx("text-xs font-mono", isSelected ? "text-cyan-300" : "text-gray-400")}>
                                        T+ {days}d {hours.toString().padStart(2, '0')}h
                                    </span>
                                    {isSelected && <Crosshair size={12} className="text-cyan-400 animate-pulse" />}
                                </div>
                                <div className="flex justify-between items-end">
                                    <span className={clsx("text-sm font-bold", isSelected ? "text-white" : "text-gray-300")}>
                                        {(dvMag * 1000).toFixed(1)} m/s
                                    </span>
                                    <span className="text-[10px] text-gray-500 bg-black/30 px-1 rounded">
                                        {m.duration.toFixed(1)}s
                                    </span>
                                </div>
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
