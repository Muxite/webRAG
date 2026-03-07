import React from "react";

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
  const paddingClass = `p-${padding}`;

  return (
    <div className={`relative ${className}`}>
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