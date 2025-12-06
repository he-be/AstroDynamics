import React, { useRef } from 'react';
import { useMissionStore } from '../stores/useMissionStore';
import { useTimeStore } from '../stores/useTimeStore';
import { Upload } from 'lucide-react';
import type { MissionManifest } from '../types/mission';

export const FileLoader: React.FC = () => {
    const loadManifest = useMissionStore((state) => state.loadManifest);
    const { setMissionDuration, setCurrentTime } = useTimeStore();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const json = JSON.parse(event.target?.result as string) as MissionManifest;

                // Validate basic structure
                if (!json.timeline || !Array.isArray(json.timeline)) {
                    throw new Error("Invalid manifest: missing timeline");
                }

                loadManifest(json);

                // Update Time Bounds
                // Assuming json.timeline keys are sorted or we find min/max
                const start = json.timeline[0].time;
                const end = json.timeline[json.timeline.length - 1].time;

                setMissionDuration(start, end);
                setCurrentTime(start);

                console.log("Loaded Mission:", json.meta.missionName);
            } catch (err) {
                console.error("Failed to parse mission file:", err);
                alert("Invalid Mission File");
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="absolute top-4 right-[240px] pointer-events-auto">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".json"
                className="hidden"
            />
            <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-[#1a1a24]/90 backdrop-blur-md border border-white/10 text-gray-300 hover:text-white px-3 py-2 rounded-lg shadow-xl flex items-center gap-2 text-sm transition-colors"
            >
                <Upload size={16} />
                <span>Load Mission</span>
            </button>
        </div>
    );
};
