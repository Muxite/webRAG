import { useState } from "react";
import { useNavigate } from "react-router";
import { supabase } from "@/lib/supabase";
import VectorBoxHeavy from "@/app/components/VectorBoxHeavy";
import VectorField from "@/app/components/VectorField";
import { getColorPalette } from "@/config/colorPalettes";

export default function Auth() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const navigate = useNavigate();

  const paletteId = "synthwave-blue";
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

  /**
   * Validates email format
   */
  const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  /**
   * Validates password strength
   */
  const isValidPassword = (password: string): { valid: boolean; error?: string } => {
    if (password.length < 6) {
      return { valid: false, error: "Password must be at least 6 characters long" };
    }
    if (password.length > 128) {
      return { valid: false, error: "Password must be less than 128 characters" };
    }
    return { valid: true };
  };

  /**
   * Creates or updates user profile in the profiles table
   */
  const ensureUserProfile = async (userId: string, userEmail: string): Promise<boolean> => {
    try {
      const { data: existingProfile, error: checkError } = await supabase
        .from("profiles")
        .select("user_id")
        .eq("user_id", userId)
        .maybeSingle();

      if (checkError && checkError.code !== "PGRST116") {
        console.error("Error checking profile:", checkError);
        return false;
      }

      if (!existingProfile) {
        const { error: insertError } = await supabase
          .from("profiles")
          .insert({
            user_id: userId,
            email: userEmail,
            daily_tick_limit: 6,
          });

        if (insertError) {
          console.error("Failed to create profile:", insertError);
          return false;
        }
      }
      return true;
    } catch (err) {
      console.error("Error ensuring profile:", err);
      return false;
    }
  };

  /**
   * Handles unified authentication - tries login first, then signup if user doesn't exist
   */
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    
    const trimmedEmail = email.trim().toLowerCase();
    const trimmedPassword = password.trim();
    
    if (!trimmedEmail) {
      setError("Email is required");
      return;
    }

    if (!isValidEmail(trimmedEmail)) {
      setError("Please enter a valid email address");
      return;
    }

    const passwordValidation = isValidPassword(trimmedPassword);
    if (!passwordValidation.valid) {
      setError(passwordValidation.error || "Invalid password");
      return;
    }
    
    setLoading(true);

    try {
      const { data: loginData, error: loginError } = await supabase.auth.signInWithPassword({
        email: trimmedEmail,
        password: trimmedPassword,
      });

      if (!loginError && loginData?.session && loginData?.user) {
        const profileCreated = await ensureUserProfile(loginData.user.id, trimmedEmail);
        if (!profileCreated) {
          console.warn("Profile creation failed, but continuing with login");
        }
        navigate("/dashboard");
        return;
      }

      if (loginError) {
        if (loginError.message.includes("Invalid login credentials") || loginError.message.includes("Email not confirmed")) {
          const { data: signupData, error: signupError } = await supabase.auth.signUp({
            email: trimmedEmail,
            password: trimmedPassword,
          });

          if (signupError) {
            if (signupError.message.includes("User already registered") || signupError.message.includes("already exists")) {
              if (loginError.message.includes("Email not confirmed")) {
                setError("Please check your email and confirm your account before logging in.");
              } else {
                setError("Invalid email or password");
              }
              setLoading(false);
              return;
            }
            
            console.error("Supabase signup error:", {
              message: signupError.message,
              status: signupError.status,
              name: signupError.name,
            });
            
            let errorMessage = "Failed to create account";
            
            if (signupError.status === 422 || signupError.name === "AuthWeakPasswordError") {
              if (signupError.message.includes("Password") || signupError.message.includes("at least 6 characters") || signupError.name === "AuthWeakPasswordError") {
                errorMessage = "Password must be at least 6 characters long.";
              } else if (signupError.message.includes("email")) {
                errorMessage = "Invalid email address format.";
              }
            } else {
              errorMessage = signupError.message || "Failed to create account";
            }
            
            throw new Error(errorMessage);
          }

          if (signupData.user) {
            if (signupData.session) {
              const profileCreated = await ensureUserProfile(signupData.user.id, trimmedEmail);
              if (!profileCreated) {
                console.warn("Profile creation failed, but continuing with signup");
              }
              navigate("/dashboard");
              return;
            } else {
              setPassword("");
              setSuccess("Account created! Please check your email to confirm your account, then try logging in again.");
              setLoading(false);
              return;
            }
          }
        } else {
          throw loginError;
        }
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
      console.error("Auth error:", err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handles password reset request
   */
  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    
    const trimmedEmail = email.trim().toLowerCase();
    
    if (!trimmedEmail) {
      setError("Please enter your email address");
      return;
    }

    if (!isValidEmail(trimmedEmail)) {
      setError("Please enter a valid email address");
      return;
    }
    
    setLoading(true);

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(trimmedEmail, {
        redirectTo: `${window.location.origin}/reset-password`,
      });

      if (error) throw error;

      setSuccess("Password reset email sent! Please check your inbox.");
      setShowForgotPassword(false);
    } catch (err: any) {
      setError(err.message || "Failed to send password reset email");
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
                EUGLENA
              </h1>
            </div>
            <p className="text-description" style={{ color: themeColors.textMuted }}>
              Automated Web Assistant
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

          {!showForgotPassword ? (
            <form onSubmit={handleAuth} className="space-y-6">
              <div>
                <label className="block text-label mb-2" style={{ color: themeColors.primary }}>
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
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
                  placeholder="user@example.com"
                  required
                  autoComplete="email"
                />
              </div>

              <div>
                <label className="block text-label mb-2" style={{ color: themeColors.primary }}>
                  Password
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
                  maxLength={128}
                  autoComplete="current-password"
                />
                <p className="mt-1 text-xs" style={{ color: themeColors.textMuted }}>
                  Password must be at least 6 characters
                </p>
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
                  {loading ? "Loading..." : "Continue"}
                </span>
                <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity" />
              </button>

              <div className="text-center">
                <button
                  type="button"
                  onClick={() => {
                    setShowForgotPassword(true);
                    setError("");
                    setSuccess("");
                  }}
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
                  Forgot password?
                </button>
              </div>
            </form>
          ) : (
            <form onSubmit={handleForgotPassword} className="space-y-6">
              <div>
                <label className="block text-label mb-2" style={{ color: themeColors.primary }}>
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
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
                  placeholder="user@example.com"
                  required
                  autoComplete="email"
                />
                <p className="mt-1 text-xs" style={{ color: themeColors.textMuted }}>
                  Enter your email address and we'll send you a password reset link
                </p>
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
                  {loading ? "Sending..." : "Send Reset Link"}
                </span>
                <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity" />
              </button>

              <div className="text-center">
                <button
                  type="button"
                  onClick={() => {
                    setShowForgotPassword(false);
                    setError("");
                    setSuccess("");
                  }}
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
          )}
        </VectorBoxHeavy>

        <div className="mt-8 w-full max-w-md">
          <div className="text-center">
            <div className="text-xs font-mono mb-2" style={{ color: themeColors.textMuted }}>
              EUGLENA <span style={{ opacity: 0.6 }}>/ WebRAG</span>
            </div>
            <div className="text-xs" style={{ color: themeColors.textMuted, opacity: 0.7 }}>
              Created by{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r" style={{
                backgroundImage: `linear-gradient(to right, ${themeColors.primary}, ${themeColors.secondary})`
              }}>
                Muk Chunpongtong
              </span>{" "}
              © 2026
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
