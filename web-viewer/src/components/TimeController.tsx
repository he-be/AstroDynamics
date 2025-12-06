import React, { useEffect, useRef } from 'react';
import { useTimeStore } from '../stores/useTimeStore';
import { Play, Pause, FastForward, Rewind } from 'lucide-react';
import clsx from 'clsx';

export const TimeController: React.FC = () => {
    const {
        isPlaying,
        currentTime,
        startTime,
        endTime,
        timeScale,
        togglePlay,
        setCurrentTime,
        setTimeScale
    } = useTimeStore();

    // Animation Loop for Time Propagation
    const requestRef = useRef<number>(0);
    const lastTimeRef = useRef<number>(0);

    const animate = (time: number) => {
        if (lastTimeRef.current !== 0 && isPlaying) {
            const deltaTime = (time - lastTimeRef.current) / 1000; // ms to seconds
            // Using direct store access to avoid closure staleness if needed, but setState handles it.
            // However, we need 'currentTime' from state efficiently. 
            // Zustand 'setState' updater is best.

            useTimeStore.setState((state) => {
                if (!state.isPlaying) return state;

                let newTime = state.currentTime + deltaTime * state.timeScale;

                // Loop or Stop at end? Stop for now.
                if (newTime >= state.endTime) {
                    newTime = state.endTime;
                    return { currentTime: newTime, isPlaying: false };
                }
                return { currentTime: newTime };
            });
        }
        lastTimeRef.current = time;
        requestRef.current = requestAnimationFrame(animate);
    };

    useEffect(() => {
        requestRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(requestRef.current);
    }, [isPlaying]); // Re-bind if playing state changes? 
    // Actually, we need the loop running always to catch updates? No, only when playing.
    // But to support correct deltaTime, we should keep the loop running or reset lastTimeRef on play.

    useEffect(() => {
        if (isPlaying) {
            lastTimeRef.current = performance.now();
            requestRef.current = requestAnimationFrame(animate);
        } else {
            cancelAnimationFrame(requestRef.current);
            lastTimeRef.current = 0;
        }
        return () => cancelAnimationFrame(requestRef.current);
    }, [isPlaying]);

    // Format Time
    // Assuming startTime is UNIX timestamp? Or just Mission Seconds? 
    // Let's assume Mission Seconds for now, relative to T=0.
    // For ISO display, we might need a separate 'epoch' prop.
    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 100);
        return `T+${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
    };

    return (
        <div className="fixed z-[100] bottom-6 left-1/2 -translate-x-1/2 w-[90%] max-w-4xl bg-gray-900/90 backdrop-blur-md border border-white/10 rounded-xl p-4 flex flex-col gap-2 text-white shadow-2xl">

            {/* Top Row: Time Display & Controls */}
            <div className="flex items-center justify-between">
                <div className="font-mono text-xl font-bold tracking-widest text-cyan-400">
                    {formatTime(currentTime)}
                </div>

                <div className="flex items-center gap-4">
                    <button onClick={() => setTimeScale(1)} className={clsx("text-xs font-bold px-2 py-1 rounded", timeScale === 1 ? "bg-cyan-600" : "bg-white/10 hover:bg-white/20")}>1x</button>
                    <button onClick={() => setTimeScale(100)} className={clsx("text-xs font-bold px-2 py-1 rounded", timeScale === 100 ? "bg-cyan-600" : "bg-white/10 hover:bg-white/20")}>100x</button>
                    <button onClick={() => setTimeScale(1000)} className={clsx("text-xs font-bold px-2 py-1 rounded", timeScale === 1000 ? "bg-cyan-600" : "bg-white/10 hover:bg-white/20")}>1000x</button>
                </div>
            </div>

            {/* Slider */}
            <input
                type="range"
                min={startTime}
                max={endTime}
                step={0.1}
                value={currentTime}
                onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500 hover:accent-cyan-400"
            />

            {/* Main Controls */}
            <div className="flex items-center justify-center gap-6 mt-1">
                <button className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-300 hover:text-white" onClick={() => setCurrentTime(startTime)}>
                    <Rewind size={20} />
                </button>

                <button
                    onClick={togglePlay}
                    className="p-3 bg-cyan-600 hover:bg-cyan-500 rounded-full shadow-lg shadow-cyan-900/50 transition-all hover:scale-105 active:scale-95"
                >
                    {isPlaying ? <Pause fill="white" size={24} /> : <Play fill="white" className="ml-1" size={24} />}
                </button>

                <button className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-300 hover:text-white" onClick={() => setCurrentTime(endTime)}>
                    <FastForward size={20} />
                </button>
            </div>
        </div>
    );
};
