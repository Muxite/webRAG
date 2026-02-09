import React, { useEffect, useRef, useState } from "react";

interface VectorFieldProps {
  spacing?: number;
  lineLength?: number;
  color?: string;
  opacity?: number;
  arrangement?: "grid" | "triangular" | "hexagonal";
  baseFieldStrength?: number; // Strength of the background rotating field (0 = pure force-based)
  noiseThreshold?: number; // Threshold for noise visibility (0-1)
  noiseScale?: number; // Scale of noise (smaller = bigger patches)
  movingSourceStrength?: number; // Strength multiplier for moving sources
  sourceSpeed?: number; // Maximum speed of moving sources
  sourceDrag?: number; // Velocity-proportional drag coefficient
  sourceAcceleration?: number; // Random acceleration magnitude
  ditherEndHeight?: number; // Height at which dithering ends (0-1, 1 = bottom of canvas)
}

// Moving sources with smooth acceleration
interface MovingSource {
  x: number;
  y: number;
  vx: number; // velocity x
  vy: number; // velocity y
  mass: number;
  nextVelocityChange: number; // time until next random velocity change
}

export default function VectorField({
  spacing = 25,
  lineLength = 18,
  color = "#06b6d4",
  opacity = 0.5,
  arrangement = "triangular",
  baseFieldStrength = 1.0,
  noiseThreshold = 0.65,
  noiseScale = 0.008,
  movingSourceStrength = 1.0,
  sourceSpeed = 0.536, // ~32px/second at 60fps (doubled from 16px/s)
  sourceDrag = 0.15,
  sourceAcceleration = 0.005,
  ditherEndHeight = 0.5, // Default to 50% of canvas height
}: VectorFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const timeRef = useRef(0);
  const movingSourcesRef = useRef<MovingSource[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Resize canvas to match window size (extended to allow dithering beyond viewport)
    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      // Extend canvas height to 3x viewport to match max dither (3.0)
      const extendedHeight = viewportHeight * 3;

      // Use viewport width (not extended) to keep sources within visible area
      canvas.width = viewportWidth * dpr;
      canvas.height = extendedHeight * dpr;
      canvas.style.width = `${viewportWidth}px`;
      canvas.style.height = `${extendedHeight}px`;
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Initialize moving sources (MORE sources, distributed across canvas)
    if (movingSourcesRef.current.length === 0) {
      const normalMass = 1600000; // Doubled from 800k for even larger influence areas
      const superMass = normalMass * 4; // 6.4M mass for super-massive sources

      const createSource = (mass: number): MovingSource => {
        const maxSpeed = sourceSpeed; // Use the sourceSpeed prop
        // Use actual window dimensions for initial spawn
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const extendedHeight = viewportHeight * 3;

        // Spawn sources distributed across the entire extended canvas
        return {
          x: Math.random() * viewportWidth, // Random X across full width
          y: Math.random() * extendedHeight, // Random Y across full extended height
          vx: (Math.random() - 0.5) * maxSpeed * 2, // Random velocity -maxSpeed to +maxSpeed
          vy: (Math.random() - 0.5) * maxSpeed * 2, // Random velocity -maxSpeed to +maxSpeed
          mass: mass,
          nextVelocityChange: Math.random() * 2000 + 1000, // 1-3 seconds
        };
      };

      // Create 12 sources total: 9 normal + 3 super-massive
      movingSourcesRef.current = [
        ...Array(9)
          .fill(null)
          .map(() => createSource(normalMass)),
        ...Array(3)
          .fill(null)
          .map(() => createSource(superMass)),
      ];
    }

    /**
     * Update moving sources with bounce physics
     * Sources bounce off top, sides, and dither end
     */
    const updateMovingSources = () => {
      // Use CSS pixel dimensions (not canvas.width which is multiplied by DPR)
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const extendedHeight = viewportHeight * 3;
      const ditherEndY = extendedHeight * ditherEndHeight; // Actual pixel Y where dither ends

      movingSourcesRef.current.forEach((source) => {
        // Add random velocity change occasionally
        source.nextVelocityChange -= 16; // Assume ~60fps, so ~16ms per frame
        if (source.nextVelocityChange <= 0) {
          const velocityBoost = sourceSpeed * 0.5; // Velocity boost is half of max speed
          source.vx += (Math.random() - 0.5) * velocityBoost;
          source.vy += (Math.random() - 0.5) * velocityBoost;
          source.nextVelocityChange =
            Math.random() * 2000 + 1000; // 1-3 seconds
        }

        // Update position
        source.x += source.vx;
        source.y += source.vy;

        // Bounce off walls with perfect elastic collision
        if (source.x < 0) {
          source.x = 0;
          source.vx = Math.abs(source.vx);
        }
        if (source.x > viewportWidth) {
          source.x = viewportWidth;
          source.vx = -Math.abs(source.vx);
        }
        if (source.y < 0) {
          source.y = 0;
          source.vy = Math.abs(source.vy);
        }
        if (source.y > ditherEndY) {
          source.y = ditherEndY;
          source.vy = -Math.abs(source.vy);
        }
      });
    };

    /**
     * Improved 2D Perlin-like noise function with smooth interpolation
     * This creates continuous noise patterns that respond well to scale changes
     */
    const noise2D = (x: number, y: number) => {
      // Smooth interpolation function (smoothstep)
      const smoothstep = (t: number) => t * t * (3 - 2 * t);

      // Hash function for pseudo-random values
      const hash = (x: number, y: number) => {
        const h =
          Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
        return h - Math.floor(h);
      };

      const scale = noiseScale;
      const nx = x * scale;
      const ny = y * scale;

      // Get integer and fractional parts
      const ix = Math.floor(nx);
      const iy = Math.floor(ny);
      const fx = nx - ix;
      const fy = ny - iy;

      // Get corner values
      const a = hash(ix, iy);
      const b = hash(ix + 1, iy);
      const c = hash(ix, iy + 1);
      const d = hash(ix + 1, iy + 1);

      // Smooth interpolation
      const sx = smoothstep(fx);
      const sy = smoothstep(fy);

      // Bilinear interpolation
      const ab = a + sx * (b - a);
      const cd = c + sx * (d - c);
      const noise1 = ab + sy * (cd - ab);

      // Add second octave for more detail (half frequency, half amplitude)
      const nx2 = nx * 2.0;
      const ny2 = ny * 2.0;
      const ix2 = Math.floor(nx2);
      const iy2 = Math.floor(ny2);
      const fx2 = nx2 - ix2;
      const fy2 = ny2 - iy2;

      const a2 = hash(ix2, iy2);
      const b2 = hash(ix2 + 1, iy2);
      const c2 = hash(ix2, iy2 + 1);
      const d2 = hash(ix2 + 1, iy2 + 1);

      const sx2 = smoothstep(fx2);
      const sy2 = smoothstep(fy2);

      const ab2 = a2 + sx2 * (b2 - a2);
      const cd2 = c2 + sx2 * (d2 - c2);
      const noise2 = ab2 + sy2 * (cd2 - ab2);

      // Combine octaves (range: 0-1.5)
      return noise1 + noise2 * 0.5;
    };

    /**
     * Calculate force from a single point mass using 1/sqrt(r) falloff
     *
     * @param px - Field point x coordinate
     * @param py - Field point y coordinate
     * @param mx - Mass location x coordinate
     * @param my - Mass location y coordinate
     * @param mass - Mass magnitude (larger = stronger force)
     * @returns Force vector { fx, fy }
     */
    const getForceFromMass = (
      px: number,
      py: number,
      mx: number,
      my: number,
      mass: number,
    ) => {
      // Displacement vector from mass to field point
      const dx = px - mx;
      const dy = py - my;

      // Distance calculation
      const rSquared = dx * dx + dy * dy;
      const r = Math.sqrt(rSquared) + 0.1; // Small epsilon to prevent division by zero

      // 1/sqrt(r) FALLOFF: F ∝ 1/r
      const forceMagnitude = mass / r;

      // Force vector components (pointing away from mass - repulsive)
      const fx = (dx / r) * forceMagnitude;
      const fy = (dy / r) * forceMagnitude;

      return { fx, fy };
    };

    /**
     * Calculate the net force (and thus field direction) at a given point
     * ALL distortion comes from moving sources ONLY
     */
    const getVectorAt = (
      x: number,
      y: number,
      time: number,
    ) => {
      // Initialize with base rotating field
      let fx = 0;
      let fy = 0;

      // BASE FIELD: Slowly rotating background pattern
      // This represents ambient forces or a "default" field configuration
      if (baseFieldStrength > 0) {
        const baseAngle =
          Math.sin(x * 0.005 + time * 0.001) *
          Math.cos(y * 0.005 - time * 0.001) *
          Math.PI;
        fx += Math.cos(baseAngle) * baseFieldStrength;
        fy += Math.sin(baseAngle) * baseFieldStrength;
      }

      // Add forces from MOVING SOURCES ONLY
      const movingSources = movingSourcesRef.current;
      movingSources.forEach((source) => {
        const force = getForceFromMass(
          x,
          y,
          source.x,
          source.y,
          source.mass,
        );
        // Apply source strength as a multiplier (default 1.0)
        fx += force.fx * movingSourceStrength;
        fy += force.fy * movingSourceStrength;
      });

      // The arrow points in the direction of the NET FORCE
      // Normalize to get direction (unit vector)
      const magnitude = Math.sqrt(fx * fx + fy * fy);

      if (magnitude > 0) {
        return {
          vx: fx / magnitude,
          vy: fy / magnitude,
        };
      }

      // Fallback if no force (shouldn't happen with base field)
      return { vx: 1, vy: 0 };
    };

    /**
     * Generate sampling points based on lattice arrangement
     */
    const generatePoints = () => {
      const points: { x: number; y: number }[] = [];

      if (arrangement === "grid") {
        // Regular Cartesian grid
        for (
          let x = spacing / 2;
          x < canvas.width;
          x += spacing
        ) {
          for (
            let y = spacing / 2;
            y < canvas.height;
            y += spacing
          ) {
            points.push({ x, y });
          }
        }
      } else if (arrangement === "triangular") {
        // Triangular lattice (equilateral triangles)
        const rowHeight = (spacing * Math.sqrt(3)) / 2;
        let row = 0;
        for (
          let y = spacing / 2;
          y < canvas.height + spacing;
          y += rowHeight
        ) {
          const offset = (row % 2) * (spacing / 2);
          for (
            let x = spacing / 2 + offset;
            x < canvas.width + spacing;
            x += spacing
          ) {
            points.push({ x, y });
          }
          row++;
        }
      } else if (arrangement === "hexagonal") {
        // Hexagonal lattice
        const hexRadius = spacing;
        const hexWidth = hexRadius * Math.sqrt(3);
        const hexHeight = hexRadius * 1.5;

        let row = 0;
        for (
          let y = hexRadius;
          y < canvas.height + hexRadius;
          y += hexHeight
        ) {
          const offset = (row % 2) * (hexWidth / 2);
          for (
            let x = hexRadius + offset;
            x < canvas.width + hexRadius;
            x += hexWidth
          ) {
            points.push({ x, y });
          }
          row++;
        }
      }

      return points;
    };

    /**
     * Draw the vector field
     */
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      timeRef.current += 1;

      // Update moving sources physics
      updateMovingSources();

      const points = generatePoints();

      // Convert opacity to hex
      const opacityHex = Math.floor(opacity * 255)
        .toString(16)
        .padStart(2, "0");

      // Draw direction indicator at each sample point
      points.forEach(({ x, y }) => {
        // Check noise value to determine visibility
        // Noise returns 0-1.5, we normalize it and use threshold to control density
        let noiseValue = noise2D(x, y) / 1.5; // Normalize to 0-1 range

        // VERTICAL DITHER: Calculate coefficient based on threshold and ditherEndHeight
        // At ditherEndHeight, even max noise (1.0) should be filtered out
        // Formula: ditherCoefficient = (1.0 - noiseThreshold) / ditherEndHeight
        const ditherCoefficient =
          ditherEndHeight > 0
            ? (1.0 - noiseThreshold) / ditherEndHeight
            : 0;
        const verticalPosition = y / canvas.height; // 0 at top, 1 at bottom
        const ditherAmount =
          verticalPosition * ditherCoefficient;
        noiseValue -= ditherAmount;

        // Only draw vectors where adjusted noise is above threshold
        // Higher threshold = fewer vectors (more sparse)
        if (noiseValue < noiseThreshold) {
          return; // Skip this vector
        }

        const { vx, vy } = getVectorAt(x, y, timeRef.current);

        // ROTATE VECTOR BY 90 DEGREES (like magnetic field around a wire)
        // This creates curved field lines instead of radial repulsion
        // (vx, vy) → (-vy, vx)
        const rotatedVx = -vy;
        const rotatedVy = vx;

        // Draw line centered at point, pointing in rotated direction
        const halfLength = lineLength / 2;
        const x1 = x - rotatedVx * halfLength;
        const y1 = y - rotatedVy * halfLength;
        const x2 = x + rotatedVx * halfLength;
        const y2 = y + rotatedVy * halfLength;

        // Draw the line
        ctx.strokeStyle = color + opacityHex;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Draw arrowhead to indicate direction
        const arrowSize = 3;
        const angle = Math.atan2(rotatedVy, rotatedVx);
        ctx.fillStyle = color + opacityHex;
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(
          x2 - arrowSize * Math.cos(angle - Math.PI / 6),
          y2 - arrowSize * Math.sin(angle - Math.PI / 6),
        );
        ctx.lineTo(
          x2 - arrowSize * Math.cos(angle + Math.PI / 6),
          y2 - arrowSize * Math.sin(angle + Math.PI / 6),
        );
        ctx.closePath();
        ctx.fill();
      });

      // Sources are invisible - no rendering
    };

    /**
     * Animation loop
     */
    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    spacing,
    lineLength,
    color,
    opacity,
    arrangement,
    baseFieldStrength,
    noiseThreshold,
    noiseScale,
    movingSourceStrength,
    sourceSpeed,
    sourceDrag,
    sourceAcceleration,
    ditherEndHeight,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
    />
  );
}