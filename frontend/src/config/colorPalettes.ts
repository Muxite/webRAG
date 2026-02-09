/**
 * COLOR PALETTE SYSTEM
 */

export interface ColorPalette {
  id: string;
  name: string;
  colors: {
    background: string;
    surface: string;
    text: string;
    textMuted: string;
    primary: string;
    secondary: string;
    accent: string;
  };
}

/**
 * All available color palettes for the application
 */
export const colorPalettes: ColorPalette[] = [
  {
    id: "neon-alley",
    name: "Neon Alley",
    colors: {
      background: "#0B0F1A",
      surface: "#141A2E",
      text: "#E6EAF2",
      textMuted: "#A8B0C2",
      primary: "#22D3EE",
      secondary: "#FF3D9A",
      accent: "#C8FF3D",
    },
  },
  {
    id: "magenta-circuit",
    name: "Magenta Circuit",
    colors: {
      background: "#0E0A14",
      surface: "#1B1026",
      text: "#E9E3F3",
      textMuted: "#B9A9D6",
      primary: "#FF4FD8",
      secondary: "#7C5CFF",
      accent: "#2AF6FF",
    },
  },
  {
    id: "synthwave-blue",
    name: "Synthwave Blue",
    colors: {
      background: "#070B14",
      surface: "#0F1730",
      text: "#E7EEFF",
      textMuted: "#9FB0D0",
      primary: "#3B82F6",
      secondary: "#A855F7",
      accent: "#FFD60A",
    },
  },
  {
    id: "toxic-teal",
    name: "Toxic Teal",
    colors: {
      background: "#061014",
      surface: "#0C1F26",
      text: "#DFF7F7",
      textMuted: "#92B8B8",
      primary: "#00F5D4",
      secondary: "#A3FF12",
      accent: "#FF2E88",
    },
  },
  {
    id: "neon-graphite",
    name: "Neon on Graphite",
    colors: {
      background: "#101218",
      surface: "#171A22",
      text: "#E5E7EB",
      textMuted: "#A1A1AA",
      primary: "#00E5FF",
      secondary: "#D946EF",
      accent: "#FFB703",
    },
  },
  {
    id: "high-contrast-blue",
    name: "High Contrast Blue",
    colors: {
      background: "#0A0E17",
      surface: "#151B2E",
      text: "#F0F4F8",
      textMuted: "#B0B8C8",
      primary: "#60D5FF",
      secondary: "#9D7EFF",
      accent: "#FFE066",
    },
  },
  {
    id: "readable-cyan",
    name: "Readable Cyan",
    colors: {
      background: "#0D1117",
      surface: "#161B22",
      text: "#E6EDF3",
      textMuted: "#8B949E",
      primary: "#58D1EB",
      secondary: "#BC8CFF",
      accent: "#F0E68C",
    },
  },
  {
    id: "soft-purple",
    name: "Soft Purple",
    colors: {
      background: "#12101A",
      surface: "#1E1B2E",
      text: "#EDE9F5",
      textMuted: "#B4A9D0",
      primary: "#9D84FF",
      secondary: "#FF88DC",
      accent: "#7FDBFF",
    },
  },
  {
    id: "balanced-teal",
    name: "Balanced Teal",
    colors: {
      background: "#0C1419",
      surface: "#172028",
      text: "#E8F2F5",
      textMuted: "#9DB5BD",
      primary: "#4DD4AC",
      secondary: "#A78BFA",
      accent: "#FFA94D",
    },
  },
];

export const getColorPalette = (id: string): ColorPalette => {
  return colorPalettes.find((p) => p.id === id) || colorPalettes[0];
};