import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { supabase } from "@/lib/supabase";
import VectorBoxHeavy from "@/app/components/VectorBoxHeavy";
import VectorField from "@/app/components/VectorField";
import { getColorPalette } from "@/config/colorPalettes";

export default function ResetPassword() {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const paletteId = "arctic-circuit";
  const themeMode: "dark" | "light" = "dark";
  const palette = getColorPalette(paletteId);
  const colors = palette.colors;

  const themeColors = {
    bg: themeMode === "dark" ? colors.background : "#F5F7FA",
    text: themeMode === "dark" ? colors.text : "#1F2937",
    textMuted: themeMode === "dark" ? colors.textMuted : "#6B7280",
    surface: themeMode === "dark" ? colors.surface : "#FFFFFF",
    boxBg: themeMode === "dark" ? `${colors.surface}D9` : "rgba(255, 255, 255, 0.85)",
    boxBgHeavy: themeMode === "dark" ? `${colors.surface}F2` : "rgba(255, 255, 255, 0.95)",
    primary: themeMode === "dark" ? colors.primary : "#3B82F6",
    secondary: themeMode === "dark" ? colors.secondary : "#8B5CF6",
    accent: themeMode === "dark" ? colors.accent : "#10B981",
  };

  useEffect(() => {
    const hashParams = new URLSearchParams(window.location.hash.substring(1));
    const accessToken = hashParams.get("access_token");
    const type = hashParams.get("type");

    if (type === "recovery" && accessToken) {
      supabase.auth.setSession({
        access_token: accessToken,
        refresh_token: hashParams.get("refresh_token") || "",
      });
    }
  }, []);

  /**
   * Handles password reset submission
   */
  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    const trimmedPassword = password.trim();
    const trimmedConfirmPassword = confirmPassword.trim();

    if (trimmedPassword.length < 6) {
      setError("Password must be at least 6 characters long");
      return;
    }

    if (trimmedPassword !== trimmedConfirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);

    try {
      const { error } = await supabase.auth.updateUser({
        password: trimmedPassword,
      });

      if (error) throw error;

      setSuccess("Password updated successfully! Redirecting to login...");
      setTimeout(() => {
        navigate("/");
      }, 2000);
    } catch (err: any) {
      setError(err.message || "Failed to update password");
      console.error("Password reset error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ 
      backgroundColor: themeColors.bg, 
      color: themeColors.text,
      '--color-primary': themeColors.primary,
      '--color-secondary': themeColors.secondary,
      '--color-text': themeColors.text,
      '--color-text-muted': themeColors.textMuted,
      '--color-accent': themeColors.accent,
    } as React.CSSProperties}>
      <VectorField 
        spacing={28}
        lineLength={14}
        color={themeColors.primary}
        opacity={themeMode === "dark" ? 0.7 : 0.21}
        arrangement="triangular"
        baseFieldStrength={1.0}
        noiseThreshold={0.4}
        noiseScale={0.008}
        movingSourceStrength={1.0}
        sourceSpeed={0.536}
        ditherEndHeight={0.5}
      />
      
      <div className="absolute inset-0 opacity-10" style={{
        backgroundImage: `linear-gradient(to right, ${themeColors.surface} 1px, transparent 1px), linear-gradient(to bottom, ${themeColors.surface} 1px, transparent 1px)`,
        backgroundSize: '4rem 4rem'
      }} />

      <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-[120px]" style={{
        backgroundColor: themeColors.secondary,
        opacity: themeMode === "dark" ? 0.05 : 0.02
      }} />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-[120px]" style={{
        backgroundColor: themeColors.primary,
        opacity: themeMode === "dark" ? 0.05 : 0.02
      }} />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen p-4">
        <VectorBoxHeavy className="w-full max-w-md" padding={8} borderColor={themeColors.primary} bgColor={themeColors.boxBgHeavy}>
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 mb-4">
              <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r" style={{
                backgroundImage: `linear-gradient(to right, ${themeColors.primary}, ${themeColors.secondary})`
              }}>
                Reset Password
              </h1>
            </div>
            <p className="text-description" style={{ color: themeColors.textMuted }}>
              Enter your new password
            </p>
          </div>

          {error && (
            <div className="mb-4 p-3 border text-metadata" style={{
              borderColor: "#EF4444",
              backgroundColor: "rgba(239, 68, 68, 0.1)",
              color: "#FCA5A5"
            }}>
              {error}
            </div>
          )}

          {success && (
            <div className="mb-4 p-3 border text-metadata" style={{
              borderColor: "#10B981",
              backgroundColor: "rgba(16, 185, 129, 0.1)",
              color: "#6EE7B7"
            }}>
              {success}
            </div>
          )}

          <form onSubmit={handleResetPassword} className="space-y-6">
            <div>
              <label className="block text-label mb-2" style={{ color: themeColors.primary }}>
                New Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full border-2 px-3 sm:px-4 py-2 focus:outline-none transition-colors text-input"
                style={{
                  backgroundColor: themeColors.surface,
                  borderColor: themeColors.primary,
                  color: themeColors.text,
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = themeColors.secondary;
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = themeColors.primary;
                }}
                placeholder="••••••••"
                required
                minLength={6}
              />
              <p className="mt-1 text-xs" style={{ color: themeColors.textMuted }}>
                Password must be at least 6 characters
              </p>
            </div>

            <div>
              <label className="block text-label mb-2" style={{ color: themeColors.primary }}>
                Confirm Password
              </label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full border-2 px-3 sm:px-4 py-2 focus:outline-none transition-colors text-input"
                style={{
                  backgroundColor: themeColors.surface,
                  borderColor: themeColors.primary,
                  color: themeColors.text,
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = themeColors.secondary;
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = themeColors.primary;
                }}
                placeholder="••••••••"
                required
                minLength={6}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full font-bold py-3 sm:py-4 px-6 transition-all disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden group"
              style={{
                backgroundColor: themeColors.primary,
                color: themeColors.bg,
              }}
            >
              <span className="relative z-10 text-button">
                {loading ? "Updating..." : "Update Password"}
              </span>
              <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity" />
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate("/")}
                className="text-sm underline transition-colors"
                style={{
                  color: themeColors.primary,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = themeColors.secondary;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = themeColors.primary;
                }}
              >
                Back to login
              </button>
            </div>
          </form>
        </VectorBoxHeavy>
      </div>
    </div>
  );
}
