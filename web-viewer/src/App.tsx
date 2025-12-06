import './App.css'


import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { SolarSystem } from './visuals/SolarSystem';
import { Spacecraft } from './visuals/Spacecraft';

import { TimeController } from './components/TimeController';
import { FrameSelector } from './components/FrameSelector';
import { ViewSettings } from './components/ViewSettings';
import { FileLoader } from './components/FileLoader';

import React, { useEffect } from 'react';
import { useMissionStore } from './stores/useMissionStore';
import { useTimeStore } from './stores/useTimeStore';
import type { MissionManifest } from './types/mission';

function App() {
  const loadManifest = useMissionStore(s => s.loadManifest);
  const { setMissionDuration, setCurrentTime } = useTimeStore();

  useEffect(() => {
    fetch('/ephemeris_default.json')
      .then(res => res.json())
      .then((data: MissionManifest) => {
        loadManifest(data);
        const start = data.timeline[0].time;
        const end = data.timeline[data.timeline.length - 1].time;
        setMissionDuration(start, end);
        setCurrentTime(start);
        console.log("Loaded Default Ephemeris");
      })
      .catch(err => console.error("Failed to load default ephemeris", err));
  }, []);
  return (
    <div className="w-full h-full bg-black relative">
      <Canvas camera={{ position: [0, 2000, 2000], fov: 45, far: 100000 }}>
        <color attach="background" args={['#050510']} />

        {/* Lights */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[10000, 0, 5000]} intensity={3.0} />

        {/* Environment */}
        <Stars radius={50000} depth={100} count={5000} factor={4} saturation={0} fade speed={1} />

        {/* Content */}
        <SolarSystem />
        <Spacecraft />

        {/* Controls */}
        <OrbitControls makeDefault minDistance={100} maxDistance={20000} />
      </Canvas>

      {/* UI Overlay */}
      <div className="fixed z-[100] top-4 left-4 text-white font-mono pointer-events-none">
        <h1 className="text-xl font-bold">OrbitViz</h1>
        <div className="text-xs text-gray-400">Jovicentric Inertial Frame</div>
      </div>

      {/* Time Controller */}
      <TimeController />

      {/* Right Controls */}
      <FileLoader />
      <FrameSelector />

      {/* Left Controls */}
      <ViewSettings />
    </div>
  )
}

export default App
