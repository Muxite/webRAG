import { useState } from "react";
import { useNavigate } from "react-router";
import { supabase } from "@/lib/supabase";
import { signupUser } from "@/api/config";
import VectorBox from "@/app/components/VectorBox";
import VectorField from "@/app/components/VectorField";

export default function Login() {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  /**
   * Handles user login form submission
   */
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;

      if (data?.session) {
        navigate("/dashboard");
      }
    } catch (err: any) {
      setError(err.message || "Failed to sign in");
      console.error("Login error:", err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handles user signup form submission
   */
  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const result = await signupUser({ email, password, name });

      if (!result.success) {
        throw new Error(result.error);
      }

      // After successful signup, switch to login mode
      setMode("login");
      setName("");
      setPassword("");
      setError("Account created! Please log in.");
    } catch (err: any) {
      setError(err.message || "Failed to create account");
      console.error("Signup error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Vector field background */}
      <VectorField spacing={25} lineLength={18} color="#06b6d4" opacity={0.5} arrangement="triangular" />
      
      {/* Cyberpunk grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f1419_1px,transparent_1px),linear-gradient(to_bottom,#0f1419_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-10" />
      
      {/* Glow effects */}
      <div className="absolute top-20 left-20 w-96 h-96 bg-purple-600 rounded-full blur-[100px] opacity-20" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-cyan-600 rounded-full blur-[100px] opacity-20" />

      <VectorBox className="w-full max-w-md relative z-10" padding={8}>
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
              EUGLENA
            </h1>
          </div>
          <p className="text-blue-300 text-description">Web-based AI Task Management</p>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            type="button"
            onClick={() => {
              setMode("login");
              setError("");
              setName("");
            }}
            className={`flex-1 py-2 px-4 text-button font-bold transition-all ${
              mode === "login"
                ? "bg-cyan-500 text-black"
                : "bg-black border border-cyan-500 text-cyan-400 hover:bg-cyan-500/10"
            }`}
          >
            Login
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("signup");
              setError("");
            }}
            className={`flex-1 py-2 px-4 text-button font-bold transition-all ${
              mode === "signup"
                ? "bg-purple-500 text-black"
                : "bg-black border border-purple-500 text-purple-400 hover:bg-purple-500/10"
            }`}
          >
            Sign Up
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 border border-red-500 bg-red-900/20 text-red-300 text-metadata">
            {error}
          </div>
        )}

        <form onSubmit={mode === "login" ? handleLogin : handleSignup} className="space-y-6">
          {mode === "signup" && (
            <div>
              <label className="block text-cyan-400 text-label mb-2">
                Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-black border-2 border-cyan-500 px-3 sm:px-4 py-2 text-white focus:outline-none focus:border-purple-500 transition-colors text-input"
                placeholder="John Doe"
                required
              />
            </div>
          )}

          <div>
            <label className="block text-cyan-400 text-label mb-2">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-black border-2 border-cyan-500 px-3 sm:px-4 py-2 text-white focus:outline-none focus:border-purple-500 transition-colors text-input"
              placeholder="user@example.com"
              required
            />
          </div>

          <div>
            <label className="block text-cyan-400 text-label mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-black border-2 border-cyan-500 px-3 sm:px-4 py-2 text-white focus:outline-none focus:border-purple-500 transition-colors text-input"
              placeholder="••••••••"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 text-black font-bold py-3 sm:py-4 px-6 hover:from-cyan-400 hover:to-purple-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden group"
          >
            <span className="relative z-10 text-button">
              {loading ? "Loading..." : mode === "login" ? "Log In" : "Sign Up"}
            </span>
            <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity" />
          </button>
        </form>
      </VectorBox>

      {/* Footer */}
      <div className="relative z-10 mt-8 w-full max-w-md">
        <div className="text-center text-xs text-cyan-400/60 font-mono">
          <div className="mb-1">
            EUGLENA <span className="text-purple-400/60">/ WebRAG</span>
          </div>
          <div className="text-gray-500">
            Created by{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
              Muk Chunpongtong
            </span>{" "}
            © 2026
          </div>
        </div>
      </div>
    </div>
  );
}