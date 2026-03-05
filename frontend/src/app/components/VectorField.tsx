import React, { useEffect, useRef, useState } from "react";

interface VectorFieldProps {
  spacing?: number;
  lineLength?: number;
  color?: string;
  opacity?: number;
  arrangement?: "grid" | "triangular" | "hexagonal";
  baseFieldStrength?: number;
  noiseThreshold?: number;
  noiseScale?: number;
  movingSourceStrength?: number;
  sourceSpeed?: number;
  sourceDrag?: number;
  sourceAcceleration?: number;
  ditherEndHeight?: number;
}

interface MovingSource {
  x: number;
  y: number;
  vx: number;
  vy: number;
  mass: number;
  nextVelocityChange: number;
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
  sourceSpeed = 0.536,
  sourceDrag = 0.15,
  sourceAcceleration = 0.005,
  ditherEndHeight = 0.5,
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

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      const extendedHeight = viewportHeight * 3;

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

    if (movingSourcesRef.current.length === 0) {
      const normalMass = 1600000;
      const superMass = normalMass * 4;

      const createSource = (mass: number): MovingSource => {
        const maxSpeed = sourceSpeed;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const extendedHeight = viewportHeight * 3;

        return {
          x: Math.random() * viewportWidth,
          y: Math.random() * extendedHeight,
          vx: (Math.random() - 0.5) * maxSpeed * 2,
          vy: (Math.random() - 0.5) * maxSpeed * 2,
          mass: mass,
          nextVelocityChange: Math.random() * 2000 + 1000,
        };
      };

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
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const extendedHeight = viewportHeight * 3;
      const ditherEndY = extendedHeight * ditherEndHeight;

      movingSourcesRef.current.forEach((source) => {
        source.nextVelocityChange -= 16;
        if (source.nextVelocityChange <= 0) {
          const velocityBoost = sourceSpeed * 0.5;
          source.vx += (Math.random() - 0.5) * velocityBoost;
          source.vy += (Math.random() - 0.5) * velocityBoost;
          source.nextVelocityChange =
            Math.random() * 2000 + 1000;
        }

        source.x += source.vx;
        source.y += source.vy;
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
      const smoothstep = (t: number) => t * t * (3 - 2 * t);

      const hash = (x: number, y: number) => {
        const h =
          Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
        return h - Math.floor(h);
      };

      const scale = noiseScale;
      const nx = x * scale;
      const ny = y * scale;

      const ix = Math.floor(nx);
      const iy = Math.floor(ny);
      const fx = nx - ix;
      const fy = ny - iy;

      const a = hash(ix, iy);
      const b = hash(ix + 1, iy);
      const c = hash(ix, iy + 1);
      const d = hash(ix + 1, iy + 1);

      const sx = smoothstep(fx);
      const sy = smoothstep(fy);

      const ab = a + sx * (b - a);
      const cd = c + sx * (d - c);
      const noise1 = ab + sy * (cd - ab);

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
      const dx = px - mx;
      const dy = py - my;

      const rSquared = dx * dx + dy * dy;
      const r = Math.sqrt(rSquared) + 0.1;

      const forceMagnitude = mass / r;

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
      let fx = 0;
      let fy = 0;

      if (baseFieldStrength > 0) {
        const baseAngle =
          Math.sin(x * 0.005 + time * 0.001) *
          Math.cos(y * 0.005 - time * 0.001) *
          Math.PI;
        fx += Math.cos(baseAngle) * baseFieldStrength;
        fy += Math.sin(baseAngle) * baseFieldStrength;
      }

      const movingSources = movingSourcesRef.current;
      movingSources.forEach((source) => {
        const force = getForceFromMass(
          x,
          y,
          source.x,
          source.y,
          source.mass,
        );
        fx += force.fx * movingSourceStrength;
        fy += force.fy * movingSourceStrength;
      });

      const magnitude = Math.sqrt(fx * fx + fy * fy);

      if (magnitude > 0) {
        return {
          vx: fx / magnitude,
          vy: fy / magnitude,
        };
      }

      return { vx: 1, vy: 0 };
    };

    /**
     * Generate sampling points based on lattice arrangement
     */
    const generatePoints = () => {
      const points: { x: number; y: number }[] = [];

      if (arrangement === "grid") {
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

      updateMovingSources();

      const points = generatePoints();

      const opacityHex = Math.floor(opacity * 255)
        .toString(16)
        .padStart(2, "0");

      points.forEach(({ x, y }) => {
        let noiseValue = noise2D(x, y) / 1.5;

        const ditherCoefficient =
          ditherEndHeight > 0
            ? (1.0 - noiseThreshold) / ditherEndHeight
            : 0;
        const verticalPosition = y / canvas.height;
        const ditherAmount =
          verticalPosition * ditherCoefficient;
        noiseValue -= ditherAmount;

        if (noiseValue < noiseThreshold) {
          return;
        }

        const { vx, vy } = getVectorAt(x, y, timeRef.current);

        const rotatedVx = -vy;
        const rotatedVy = vx;

        const halfLength = lineLength / 2;
        const x1 = x - rotatedVx * halfLength;
        const y1 = y - rotatedVy * halfLength;
        const x2 = x + rotatedVx * halfLength;
        const y2 = y + rotatedVy * halfLength;

        ctx.strokeStyle = color + opacityHex;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

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