import React, { useEffect, useRef, useState } from "react";

interface PolygonBoxProps {
  children: React.ReactNode;
  className?: string;
  density?: number; // How many points per side (default: 20)
  spikiness?: number; // How much variation in the edge (default: 15)
  margin?: number; // SVG margin from edge (default: 1)
  padding?: number; // Content padding in Tailwind units (default: 5)
  borderColor?: string; // Border color (default: #06b6d4)
  borderThickness?: number; // Border thickness (default: 1)
  fillColor?: string; // Fill color (default: transparent)
}

export default function PolygonBox({
  children,
  className = "",
  density = 20,
  spikiness = 15,
  margin = 1,
  padding = 5,
  borderColor = "#06b6d4",
  borderThickness = 1,
  fillColor = "rgba(0, 0, 0, 0)",
}: PolygonBoxProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [lines, setLines] = useState<string[]>([]);
  const animationRef = useRef<number>();
  const timeRef = useRef(0);

  // Check if this is a compact info box
  const isCompact = className.includes("compact-info");
  const paddingClass = `p-${padding}`;

  // Simple noise function (Perlin-like approximation)
  const noise = (x: number, y: number, time: number): number => {
    const scale = 0.02;
    const timeScale = 0.0005;
    const nx = x * scale + time * timeScale;
    const ny = y * scale + time * timeScale * 0.7;
    
    // Multiple octaves of sine waves for pseudo-noise
    const noise1 = Math.sin(nx * 2.1 + ny * 1.3) * 0.5;
    const noise2 = Math.sin(nx * 3.7 - ny * 2.9) * 0.3;
    const noise3 = Math.sin(nx * 5.3 + ny * 4.7) * 0.2;
    
    return noise1 + noise2 + noise3;
  };

  // Generate 4 separate noisy lines that extend beyond corners
  const generateExtendingLines = (width: number, height: number, time: number) => {
    const lines: string[] = [];
    const extension = 20; // How much lines extend beyond corners
    
    // If density is 0, just use straight lines
    if (density === 0) {
      lines.push(`M ${-extension} ${margin} L ${width / 2} ${margin}`); // Top left to center
      lines.push(`M ${width / 2} ${margin} L ${width + extension} ${margin}`); // Top center to right
      lines.push(`M ${width - margin} ${-extension} L ${width - margin} ${height / 2}`); // Right top to center
      lines.push(`M ${width - margin} ${height / 2} L ${width - margin} ${height + extension}`); // Right center to bottom
      lines.push(`M ${width + extension} ${height - margin} L ${width / 2} ${height - margin}`); // Bottom right to center
      lines.push(`M ${width / 2} ${height - margin} L ${-extension} ${height - margin}`); // Bottom center to left
      lines.push(`M ${margin} ${height + extension} L ${margin} ${height / 2}`); // Left bottom to center
      lines.push(`M ${margin} ${height / 2} L ${margin} ${-extension}`); // Left center to top
      return lines;
    }

    const generateNoisyLine = (
      startX: number, 
      startY: number, 
      endX: number, 
      endY: number, 
      normalX: number, 
      normalY: number
    ) => {
      const points: [number, number][] = [];
      const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
      const pointCount = Math.max(3, Math.floor(length / 10)); // At least 3 points per line
      
      for (let i = 0; i <= pointCount; i++) {
        const t = i / pointCount;
        const baseX = startX + (endX - startX) * t;
        const baseY = startY + (endY - startY) * t;
        
        // Apply noise perpendicular to the line
        const noiseValue = noise(baseX, baseY, time) * spikiness;
        const x = baseX + normalX * noiseValue;
        const y = baseY + normalY * noiseValue;
        
        points.push([x, y]);
      }
      
      // Convert to SVG path
      const pathData = points.map((p, i) => 
        `${i === 0 ? 'M' : 'L'} ${p[0]} ${p[1]}`
      ).join(' ');
      
      return pathData;
    };

    // Top line: extends left and right
    lines.push(generateNoisyLine(
      -extension, margin,
      width + extension, margin,
      0, -1 // Normal pointing up
    ));

    // Right line: extends up and down
    lines.push(generateNoisyLine(
      width - margin, -extension,
      width - margin, height + extension,
      1, 0 // Normal pointing right
    ));

    // Bottom line: extends left and right
    lines.push(generateNoisyLine(
      width + extension, height - margin,
      -extension, height - margin,
      0, 1 // Normal pointing down
    ));

    // Left line: extends up and down
    lines.push(generateNoisyLine(
      margin, height + extension,
      margin, -extension,
      -1, 0 // Normal pointing left
    ));

    return lines;
  };

  // Update dimensions when container size changes
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

  // Animate the polygon
  useEffect(() => {
    if (dimensions.width === 0 || dimensions.height === 0) return;

    const animate = () => {
      timeRef.current += 1;
      const newLines = generateExtendingLines(dimensions.width, dimensions.height, timeRef.current);
      setLines(newLines);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [dimensions, density, spikiness, margin]);

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* SVG Border Lines */}
      <svg
        className="absolute inset-0 pointer-events-none"
        width={dimensions.width}
        height={dimensions.height}
        style={{ overflow: "visible" }}
      >
        {/* Semi-transparent fill rect */}
        <rect
          x={margin}
          y={margin}
          width={dimensions.width - margin * 2}
          height={dimensions.height - margin * 2}
          fill={fillColor}
        />
        
        {/* Render each line with glow layers */}
        {lines.map((pathData, index) => (
          <g key={index}>
            {/* Outer glow */}
            <path
              d={pathData}
              fill="none"
              stroke={borderColor}
              strokeWidth={borderThickness * 3}
              style={{
                filter: "blur(8px)",
                opacity: 0.6,
              }}
            />
            
            {/* Mid glow */}
            <path
              d={pathData}
              fill="none"
              stroke={borderColor}
              strokeWidth={borderThickness * 2}
              style={{
                filter: "blur(4px)",
                opacity: 0.7,
              }}
            />
            
            {/* Inner glow */}
            <path
              d={pathData}
              fill="none"
              stroke={borderColor}
              strokeWidth={borderThickness * 1.5}
              style={{
                filter: "blur(2px)",
                opacity: 0.9,
              }}
            />
            
            {/* Core sharp line */}
            <path
              d={pathData}
              fill="none"
              stroke={borderColor}
              strokeWidth={borderThickness}
              style={{
                opacity: 1,
              }}
            />
          </g>
        ))}
      </svg>

      {/* Content with transparent background */}
      <div className={`relative z-10 ${paddingClass}`}>
        <div className="relative z-10">{children}</div>
      </div>

      {/* Corner accents with glow */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
    </div>
  );
}