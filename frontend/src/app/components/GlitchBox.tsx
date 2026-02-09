import React from "react";

interface GlitchBoxProps {
  children: React.ReactNode;
  className?: string;
}

export default function GlitchBox({ children, className = "" }: GlitchBoxProps) {
  // Check if this is a compact info box
  const isCompact = className.includes('compact-info');
  const padding = isCompact ? 'p-3' : 'p-8';
  
  return (
    <div className={`relative ${className}`}>
      {/* Cloud-like fuzzy border effect - Gaussian clusters around entire perimeter */}
      <div className="absolute inset-0 pointer-events-none overflow-visible">
        {/* Outer layer 1 - Cyan - Large Gaussian blur from edge centers */}
        <div
          className="absolute -inset-[50px] opacity-90"
          style={{
            background: `
              radial-gradient(ellipse 800px 120px at 50% 0%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 800px 120px at 50% 100%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 120px 800px at 0% 50%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 120px 800px at 100% 50%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.6) 30%, transparent 70%)
            `,
            filter: "blur(30px)",
            animation: "cloudMove1 4s ease-in-out infinite",
          }}
        />

        {/* Outer layer 2 - Purple - Large Gaussian blur from edge centers */}
        <div
          className="absolute -inset-[48px] opacity-85"
          style={{
            background: `
              radial-gradient(ellipse 800px 110px at 50% 0%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 800px 110px at 50% 100%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 110px 800px at 0% 50%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.6) 30%, transparent 70%),
              radial-gradient(ellipse 110px 800px at 100% 50%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.6) 30%, transparent 70%)
            `,
            filter: "blur(32px)",
            animation: "cloudMove2 5s ease-in-out infinite",
          }}
        />

        {/* Mid layer 1 - Mixed cyan/purple - Medium Gaussian */}
        <div
          className="absolute -inset-[40px] opacity-95"
          style={{
            background: `
              radial-gradient(ellipse 600px 100px at 50% 0%, rgba(6, 182, 212, 1) 0%, rgba(87, 133, 229, 0.8) 25%, transparent 65%),
              radial-gradient(ellipse 600px 100px at 50% 100%, rgba(168, 85, 247, 1) 0%, rgba(87, 133, 229, 0.8) 25%, transparent 65%),
              radial-gradient(ellipse 100px 600px at 0% 50%, rgba(6, 182, 212, 1) 0%, rgba(87, 133, 229, 0.8) 25%, transparent 65%),
              radial-gradient(ellipse 100px 600px at 100% 50%, rgba(168, 85, 247, 1) 0%, rgba(87, 133, 229, 0.8) 25%, transparent 65%)
            `,
            filter: "blur(25px)",
            animation: "cloudMove3 3.5s ease-in-out infinite",
          }}
        />

        {/* Mid layer 2 - Alternating colors for variation */}
        <div
          className="absolute -inset-[30px] opacity-90"
          style={{
            background: `
              radial-gradient(ellipse 500px 80px at 50% 0%, rgba(168, 85, 247, 1) 0%, rgba(120, 119, 198, 0.7) 25%, transparent 60%),
              radial-gradient(ellipse 500px 80px at 50% 100%, rgba(6, 182, 212, 1) 0%, rgba(120, 119, 198, 0.7) 25%, transparent 60%),
              radial-gradient(ellipse 80px 500px at 0% 50%, rgba(168, 85, 247, 1) 0%, rgba(120, 119, 198, 0.7) 25%, transparent 60%),
              radial-gradient(ellipse 80px 500px at 100% 50%, rgba(6, 182, 212, 1) 0%, rgba(120, 119, 198, 0.7) 25%, transparent 60%)
            `,
            filter: "blur(20px)",
            animation: "cloudMove4 3s ease-in-out infinite",
          }}
        />

        {/* Inner layer - Bright core glow */}
        <div
          className="absolute -inset-[20px] opacity-95"
          style={{
            background: `
              radial-gradient(ellipse 400px 70px at 50% 0%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.8) 20%, transparent 55%),
              radial-gradient(ellipse 400px 70px at 50% 100%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.8) 20%, transparent 55%),
              radial-gradient(ellipse 70px 400px at 0% 50%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.8) 20%, transparent 55%),
              radial-gradient(ellipse 70px 400px at 100% 50%, rgba(6, 182, 212, 1) 0%, rgba(6, 182, 212, 0.8) 20%, transparent 55%)
            `,
            filter: "blur(15px)",
            animation: "cloudMove1 3.2s ease-in-out infinite reverse",
          }}
        />

        {/* Innermost bright edge */}
        <div
          className="absolute -inset-[10px] opacity-100"
          style={{
            background: `
              radial-gradient(ellipse 300px 50px at 50% 0%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.9) 15%, transparent 50%),
              radial-gradient(ellipse 300px 50px at 50% 100%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.9) 15%, transparent 50%),
              radial-gradient(ellipse 50px 300px at 0% 50%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.9) 15%, transparent 50%),
              radial-gradient(ellipse 50px 300px at 100% 50%, rgba(168, 85, 247, 1) 0%, rgba(168, 85, 247, 0.9) 15%, transparent 50%)
            `,
            filter: "blur(10px)",
            animation: "cloudMove2 2.8s ease-in-out infinite reverse",
          }}
        />
      </div>

      {/* Content with transparent center allowing fuzzy border to show */}
      <div className={`relative z-10 bg-black/50 ${padding} backdrop-blur-sm`}>
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