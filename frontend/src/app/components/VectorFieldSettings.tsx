import React, { useState } from "react";
import { Settings, ChevronDown, ChevronUp } from "lucide-react";

interface VectorFieldSettingsProps {
  spacing: number;
  lineLength: number;
  opacity: number;
  arrangement: "grid" | "triangular" | "hexagonal";
  baseFieldStrength: number;
  noiseThreshold: number;
  noiseScale: number;
  movingSourceStrength: number;
  sourceSpeed: number;
  ditherEndHeight: number;
  onSpacingChange: (value: number) => void;
  onLineLengthChange: (value: number) => void;
  onOpacityChange: (value: number) => void;
  onArrangementChange: (value: "grid" | "triangular" | "hexagonal") => void;
  onBaseFieldStrengthChange: (value: number) => void;
  onNoiseThresholdChange: (value: number) => void;
  onNoiseScaleChange: (value: number) => void;
  onMovingSourceStrengthChange: (value: number) => void;
  onSourceSpeedChange: (value: number) => void;
  onDitherEndHeightChange: (value: number) => void;
}

export default function VectorFieldSettings({
  spacing,
  lineLength,
  opacity,
  arrangement,
  baseFieldStrength,
  noiseThreshold,
  noiseScale,
  movingSourceStrength,
  sourceSpeed,
  ditherEndHeight,
  onSpacingChange,
  onLineLengthChange,
  onOpacityChange,
  onArrangementChange,
  onBaseFieldStrengthChange,
  onNoiseThresholdChange,
  onNoiseScaleChange,
  onMovingSourceStrengthChange,
  onSourceSpeedChange,
  onDitherEndHeightChange,
}: VectorFieldSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleDitherChange = (value: number) => {
    onDitherEndHeightChange(value);
  };

  const handleToggle = () => {
    console.log("Field button clicked, isOpen:", isOpen);
    setIsOpen(!isOpen);
  };

  return (
    <div className="fixed top-4 right-4 z-[100] pointer-events-auto">
      <button
        onClick={handleToggle}
        className="flex items-center gap-1.5 px-3 py-1.5 border border-cyan-500 text-cyan-400 hover:bg-cyan-500/10 transition-colors text-xs font-mono"
        style={{
          boxShadow: "0 0 10px rgba(6, 182, 212, 0.3)",
          backgroundColor: "rgba(0, 0, 0, 0.85)",
        }}
      >
        <Settings className="w-4 h-4" />
        <span>FIELD</span>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-64 border-2 border-cyan-500 p-3 space-y-3 bg-black/80"
          style={{
            boxShadow: "0 0 20px rgba(6, 182, 212, 0.4)",
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-3 h-3 text-cyan-400" />
            <h3 className="text-cyan-400 text-small-label">VECTOR_FIELD</h3>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              ARRANGEMENT
            </label>
            <select
              value={arrangement}
              onChange={(e) => onArrangementChange(e.target.value as any)}
              className="w-full bg-black border border-cyan-500 px-1.5 py-0.5 text-cyan-400 text-small-label focus:outline-none focus:border-purple-500"
            >
              <option value="grid">GRID</option>
              <option value="triangular">TRIANGULAR</option>
              <option value="hexagonal">HEXAGONAL</option>
            </select>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              SPACING: {spacing}px
            </label>
            <input
              type="range"
              min="15"
              max="60"
              value={spacing}
              onChange={(e) => onSpacingChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              LINE_LENGTH: {lineLength}px
            </label>
            <input
              type="range"
              min="10"
              max="80"
              value={lineLength}
              onChange={(e) => onLineLengthChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              OPACITY: {Math.round(opacity * 100)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={opacity}
              onChange={(e) => onOpacityChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              SOURCE_STRENGTH: {movingSourceStrength.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.0"
              max="20.0"
              step="0.1"
              value={movingSourceStrength}
              onChange={(e) => onMovingSourceStrengthChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="text-small-label text-gray-500 mt-0.5">
              Multiplier for moving source forces (0.0-20.0)
            </div>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              SOURCE_SPEED: {sourceSpeed.toFixed(4)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={sourceSpeed}
              onChange={(e) => onSourceSpeedChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="text-small-label text-gray-500 mt-0.5">
              Speed of source movement (0.001-0.1)
            </div>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              NOISE_THRESHOLD: {(noiseThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={noiseThreshold}
              onChange={(e) => onNoiseThresholdChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="text-small-label text-gray-500 mt-0.5">
              Lower = denser field, Higher = sparser field
            </div>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              NOISE_SCALE: {noiseScale.toFixed(4)}
            </label>
            <input
              type="range"
              min="0.0001"
              max="0.05"
              step="0.0001"
              value={noiseScale}
              onChange={(e) => onNoiseScaleChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="text-small-label text-gray-500 mt-0.5">
              Lower = Bigger patches
            </div>
          </div>

          <div>
            <label className="block text-purple-400 text-small-label mb-1">
              DITHER_END_HEIGHT: {ditherEndHeight.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.0"
              max="3.0"
              step="0.05"
              value={ditherEndHeight}
              onChange={(e) => handleDitherChange(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="text-small-label text-gray-500 mt-0.5">
              Height where vectors disappear (0.0-3.0, 1.0=bottom)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}