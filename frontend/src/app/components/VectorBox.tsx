import React, { useRef } from "react";

interface VectorBoxProps {
  children: React.ReactNode;
  className?: string;
  padding?: number;
  borderColor?: string;
  glowIntensity?: "low" | "medium" | "high";
  bgColor?: string;
}

export default function VectorBox({
  children,
  className = "",
  padding = 6,
  borderColor = "#06b6d4",
  glowIntensity = "medium",
  bgColor = "rgba(0, 0, 0, 0.85)",
}: VectorBoxProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const paddingClass = `p-${padding}`;
  
  const glowStyles = {
    low: "shadow-[0_0_10px_rgba(6,182,212,0.3)]",
    medium: "shadow-[0_0_20px_rgba(6,182,212,0.5)]",
    high: "shadow-[0_0_30px_rgba(6,182,212,0.7)]",
  };
  
  const glowClass = glowStyles[glowIntensity];

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* Main container with gradient border */}
      <div
        className={`relative border ${paddingClass}`}
        style={{
          borderColor: borderColor,
          background: bgColor,
          boxShadow: `0 0 10px ${borderColor}25, inset 0 0 10px ${borderColor}15`,
        }}
      >
        {/* Content */}
        <div className="relative z-10">{children}</div>
      </div>
    </div>
  );
}