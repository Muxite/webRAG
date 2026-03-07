import React from "react";

interface VectorBoxHeavyProps {
  children: React.ReactNode;
  className?: string;
  padding?: number;
  borderColor?: string;
  glowIntensity?: "low" | "medium" | "high";
  bgColor?: string;
}

export default function VectorBoxHeavy({
  children,
  className = "",
  padding = 6,
  borderColor = "#06b6d4",
  glowIntensity = "high",
  bgColor = "rgba(0, 0, 0, 0.9)",
}: VectorBoxHeavyProps) {
  const paddingClass = `p-${padding}`;

  return (
    <div className={`relative ${className}`}>
      {/* Main container with enhanced gradient border */}
      <div
        className={`relative border-2 ${paddingClass}`}
        style={{
          borderColor: borderColor,
          background: bgColor,
          boxShadow: `0 0 15px ${borderColor}30, inset 0 0 15px ${borderColor}15`,
        }}
      >
        {/* Content */}
        <div className="relative z-10">{children}</div>
      </div>
    </div>
  );
}