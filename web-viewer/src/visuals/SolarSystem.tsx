import React from 'react';
import { Body } from './Body';

export const SolarSystem: React.FC = () => {
    // Constants (km)
    const R_JUPITER = 71492;
    const R_IO = 1821;
    const R_EUROPA = 1560;
    const R_GANYMEDE = 2634;
    const R_CALLISTO = 2410;

    // Scale for visibility (1 unit = 1 km is too big for Three default cam, 
    // but let's stick to 1:1 and move camera far away first).
    // Actually, typical WebGL depth precision suggests scaling down.
    // Let's use 1 unit = 1000 km.
    const UNIT_SCALE = 0.001;

    return (
        <group scale={[UNIT_SCALE, UNIT_SCALE, UNIT_SCALE]}>
            {/* Jupiter */}
            <Body
                name="Jupiter"
                radius={R_JUPITER}
                color="#d4a373"
                position={[0, 0, 0]}
            />

            {/* Moons (Mock Positions for Visualization Check) */}
            <Body
                name="Io"
                radius={R_IO}
                color="#f4e409"
                position={[421700, 0, 0]}
                scaleFactor={5} // Exaggerate moons for now
            />
            <Body
                name="Europa"
                radius={R_EUROPA}
                color="#a67c52"
                position={[671100, 0, 0]}
                scaleFactor={5}
            />
            <Body
                name="Ganymede"
                radius={R_GANYMEDE}
                color="#706f6f"
                position={[1070400, 0, 0]}
                scaleFactor={5}
            />
            <Body
                name="Callisto"
                radius={R_CALLISTO}
                color="#5c5c5c"
                position={[1882700, 0, 0]}
                scaleFactor={5}
            />
        </group>
    );
};
