import React from 'react';

interface BodyProps {
    name: string;
    radius: number; // km
    color: string;
    position: [number, number, number];
    scaleFactor?: number; // Visual scaling for visibility
}

export const Body: React.FC<BodyProps> = ({ name, radius, color, position, scaleFactor = 1.0 }) => {
    // Real radii are too small compared to distances. 
    // We might need significant visual scaling for icons/dots if zoomed out.
    // For now, render direct size * scaleFactor.

    return (
        <mesh position={position} userData={{ name }}>
            <sphereGeometry args={[radius * scaleFactor, 32, 32]} />
            <meshStandardMaterial color={color} />
            {/* Optional Label could go here */}
        </mesh>
    );
};
