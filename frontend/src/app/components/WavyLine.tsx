import React, { useEffect, useRef, useState } from "react";

interface WavyLineProps {
  orientation?: "horizontal" | "vertical";
  color?: string;
  thickness?: number;
  density?: number;
  amplitude?: number;
  className?: string;
}

export default function WavyLine({
  orientation = "horizontal",
  color = "rgba(6, 182, 212, 1)",
  thickness = 2,
  density = 30,
  amplitude = 8,
  className = "",
}: WavyLineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [pathData, setPathData] = useState("");
  const animationRef = useRef<number>();
  const timeRef = useRef(0);

  // Simple noise function
  const noise = (x: number, time: number): number => {
    const scale = 0.015;
    const timeScale = 0.0003;
    const nx = x * scale + time * timeScale;
    
    const noise1 = Math.sin(nx * 2.3) * 0.5;
    const noise2 = Math.sin(nx * 4.1 - time * 0.0002) * 0.3;
    const noise3 = Math.sin(nx * 6.7 + time * 0.0001) * 0.2;
    
    return noise1 + noise2 + noise3;
  };

  // Generate wavy path
  const generateWavyPath = (width: number, height: number, time: number): string => {
    const points: string[] = [];
    
    if (orientation === "horizontal") {
      const centerY = height / 2;
      
      for (let i = 0; i <= density; i++) {
        const t = i / density;
        const x = t * width;
        const noiseValue = noise(x, time) * amplitude;
        const y = centerY + noiseValue;
        
        if (i === 0) {
          points.push(`M ${x} ${y}`);
        } else {
          points.push(`L ${x} ${y}`);
        }
      }
    } else {
      // Vertical orientation
      const centerX = width / 2;
      
      for (let i = 0; i <= density; i++) {
        const t = i / density;
        const y = t * height;
        const noiseValue = noise(y, time) * amplitude;
        const x = centerX + noiseValue;
        
        if (i === 0) {
          points.push(`M ${x} ${y}`);
        } else {
          points.push(`L ${x} ${y}`);
        }
      }
    }
    
    return points.join(" ");
  };

  // Update dimensions
  useEffect(() => {
    if (!containerRef.current) return;

    const updateDimensions = () => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) {
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);

    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Animate the wavy line
  useEffect(() => {
    if (dimensions.width === 0 || dimensions.height === 0) return;

    const animate = () => {
      timeRef.current += 1;
      const path = generateWavyPath(dimensions.width, dimensions.height, timeRef.current);
      setPathData(path);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [dimensions, density, amplitude, orientation]);

  const height = orientation === "horizontal" ? `${amplitude * 2 + thickness}px` : "100%";
  const width = orientation === "horizontal" ? "100%" : `${amplitude * 2 + thickness}px`;

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        width,
        height,
        position: "relative",
      }}
    >
      <svg
        width={dimensions.width}
        height={dimensions.height}
        style={{ position: "absolute", top: 0, left: 0 }}
      >
        {/* Outer glow */}
        <path
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth={thickness + 4}
          style={{
            filter: "blur(8px)",
            opacity: 0.4,
          }}
        />
        
        {/* Mid glow - purple tint */}
        <path
          d={pathData}
          fill="none"
          stroke="rgba(168, 85, 247, 0.8)"
          strokeWidth={thickness + 2}
          style={{
            filter: "blur(4px)",
            opacity: 0.6,
          }}
        />
        
        {/* Sharp core */}
        <path
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth={thickness}
          style={{
            opacity: 1,
          }}
        />
      </svg>
    </div>
  );
}
