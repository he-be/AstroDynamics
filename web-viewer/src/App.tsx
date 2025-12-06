import './App.css'


import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { SolarSystem } from './visuals/SolarSystem';

import { TimeController } from './components/TimeController';
import { FrameSelector } from './components/FrameSelector';

function App() {
  return (
    <div className="w-full h-full bg-black relative">
      <Canvas camera={{ position: [0, 2000, 2000], fov: 45, far: 100000 }}>
        <color attach="background" args={['#050510']} />

        {/* Lights */}
        <ambientLight intensity={0.1} />
        <pointLight position={[0, 0, 0]} intensity={2.0} />

        {/* Environment */}
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

        {/* Content */}
        <SolarSystem />

        {/* Controls */}
        <OrbitControls makeDefault minDistance={100} maxDistance={20000} />
      </Canvas>

      {/* UI Overlay */}
      <div className="absolute top-4 left-4 text-white font-mono pointer-events-none">
        <h1 className="text-xl font-bold">OrbitViz</h1>
        <div className="text-xs text-gray-400">Jovicentric Inertial Frame</div>
      </div>

      {/* Time Controller */}
      <TimeController />

      {/* Frame Selector */}
      <FrameSelector />
    </div>
  )
}

export default App
