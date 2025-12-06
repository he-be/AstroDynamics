import React from 'react';
import { Html } from '@react-three/drei';
import { useVisualizationStore } from '../stores/useVisualizationStore';

interface BodyProps {
    name: string;
    radius: number; // km
    color: string;
    position: [number, number, number];
    scaleFactor?: number; // Visual scaling for visibility
}

export const Body: React.FC<BodyProps> = ({ name, radius, color, position, scaleFactor = 1.0 }) => {
    const showOutlines = useVisualizationStore(s => s.showOutlines);

    return (
        <group position={position}>
            <mesh userData={{ name }}>
                <sphereGeometry args={[radius * scaleFactor, 32, 32]} />
                <meshStandardMaterial color={color} />
            </mesh>

            {showOutlines && (
                <Html position={[0, 0, 0]} zIndexRange={[50, 0]}>
                    <div className="relative pointer-events-none">
                        {/* Dot: Centered on [0,0,0] */}
                        <div className="absolute top-0 left-0 w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,1)]"
                            style={{ transform: 'translate(-50%, -50%)' }} />

                        {/* Label: Centered horizontally, pushed down */}
                        <div className="absolute top-3 left-0 text-[10px] text-white font-mono whitespace-nowrap bg-black/50 px-1 rounded backdrop-blur-sm"
                            style={{ transform: 'translate(-50%, 0)' }}>
                            {name}
                        </div>
                    </div>
                </Html>
            )}
        </group>
    );
};
