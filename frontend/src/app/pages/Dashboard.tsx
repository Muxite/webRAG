import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { supabase } from "@/lib/supabase";
import { LogOut, Send, RefreshCw, Palette, Github } from "lucide-react";
import VectorBox from "@/app/components/VectorBox";
import VectorBoxHeavy from "@/app/components/VectorBoxHeavy";
import VectorField from "@/app/components/VectorField";
import VectorFieldSettings from "@/app/components/VectorFieldSettings";
import TaskContainer from "@/app/components/TaskContainer";
import { motion } from "motion/react";
import { colorPalettes, getColorPalette } from "@/config/colorPalettes";
import {
  fetchSystemInfo,
  fetchUserStats,
  fetchTasks,
  submitTask,
  deleteTask,
  POLLING_INTERVAL,
  type SystemInfo,
  type UserStats,
  type Task,
} from "@/api/config";

export default function Dashboard() {
  const [user, setUser] = useState<any>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [mandate, setMandate] = useState("");
  const [maxTicks, setMaxTicks] = useState<number>(10);
  const [loading, setLoading] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  const [showReconnect, setShowReconnect] = useState(false);
  
  const [paletteId, setPaletteId] = useState<string>("neon-alley");
  const [showPaletteMenu, setShowPaletteMenu] = useState(false);
  const [themeMode, setThemeMode] = useState<"dark" | "light">("dark");
  
  const [fieldSpacing, setFieldSpacing] = useState(28);
  const [fieldLineLength, setFieldLineLength] = useState(14);
  const [fieldOpacity, setFieldOpacity] = useState(0.7);
  const [fieldArrangement, setFieldArrangement] = useState<"grid" | "triangular" | "hexagonal">("triangular");
  const [baseFieldStrength, setBaseFieldStrength] = useState(1.0);
  const [noiseThreshold, setNoiseThreshold] = useState(0.4);
  const [noiseScale, setNoiseScale] = useState(0.008);
  const [movingSourceStrength, setMovingSourceStrength] = useState(1.0);
  const [sourceSpeed, setSourceSpeed] = useState(0.536);
  const [ditherEndHeight, setDitherEndHeight] = useState(0.5);
  
  const navigate = useNavigate();
  
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
    checkAuth();
  }, []);

  useEffect(() => {
    if (user) {
      loadSystemInfo();
      loadUserStats();
      loadTasks();
      
      const interval = setInterval(() => {
        loadTasks();
        loadUserStats();
      }, POLLING_INTERVAL);
      
      return () => clearInterval(interval);
    }
  }, [user]);

  /**
   * Checks authentication status and redirects to login if not authenticated
   */
  const checkAuth = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        navigate("/");
        return;
      }

      setUser(session.user);
    } catch (error) {
      setShowReconnect(true);
    }
  };

  /**
   * Reloads the page to reconnect to Supabase
   */
  const handleReconnectSupabase = () => {
    window.location.reload();
  };

  /**
   * Fetches and updates system information
   */
  const loadSystemInfo = async () => {
    const data = await fetchSystemInfo();
    setSystemInfo(data);
  };

  /**
   * Fetches and updates user statistics
   */
  const loadUserStats = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) return;

      const data = await fetchUserStats(session.access_token, session.user.email);
      setUserStats(data);
    } catch (error) {
      console.error("Error loading user stats:", error);
    }
  };

  /**
   * Fetches and updates task list
   */
  const loadTasks = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) return;

      const data = await fetchTasks(session.access_token);
      setTasks(data);
    } catch (error) {
      console.error("Error loading tasks:", error);
    }
  };

  /**
   * Handles task submission form
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!mandate.trim() || maxTicks <= 0) return;

    setLoading(true);
    setSubmitMessage(null);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        return;
      }

      const response = await submitTask(session.access_token, { mandate, maxTicks });

      if (response) {
        setSubmitMessage(`Task received: ${response.correlation_id}`);
        setMandate("");
        setMaxTicks(10);
        await loadTasks();
        await loadUserStats();
      } else {
        setSubmitMessage("Task submission failed");
      }
    } catch (error) {
      setSubmitMessage("Task submission failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteTask = async (taskId: string) => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        return;
      }

      const success = await deleteTask(session.access_token, taskId);
      if (success) {
        await loadTasks();
      }
    } catch (error) {
      console.error("Error deleting task:", error);
    }
  };

  /**
   * Signs out user and redirects to login page
   */
  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate("/");
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
      {/* Vector field background */}
      <VectorField 
        spacing={fieldSpacing}
        lineLength={fieldLineLength}
        color={themeColors.primary}
        opacity={themeMode === "dark" ? fieldOpacity : fieldOpacity * 0.3}
        arrangement={fieldArrangement}
        baseFieldStrength={baseFieldStrength}
        noiseThreshold={noiseThreshold}
        noiseScale={noiseScale}
        movingSourceStrength={movingSourceStrength}
        sourceSpeed={sourceSpeed}
        ditherEndHeight={ditherEndHeight}
      />
      
      {/* Vector Field Settings */}
      <VectorFieldSettings
        spacing={fieldSpacing}
        lineLength={fieldLineLength}
        opacity={fieldOpacity}
        arrangement={fieldArrangement}
        baseFieldStrength={baseFieldStrength}
        noiseThreshold={noiseThreshold}
        noiseScale={noiseScale}
        movingSourceStrength={movingSourceStrength}
        sourceSpeed={sourceSpeed}
        ditherEndHeight={ditherEndHeight}
        onSpacingChange={setFieldSpacing}
        onLineLengthChange={setFieldLineLength}
        onOpacityChange={setFieldOpacity}
        onArrangementChange={setFieldArrangement}
        onBaseFieldStrengthChange={setBaseFieldStrength}
        onNoiseThresholdChange={setNoiseThreshold}
        onNoiseScaleChange={setNoiseScale}
        onMovingSourceStrengthChange={setMovingSourceStrength}
        onSourceSpeedChange={setSourceSpeed}
        onDitherEndHeightChange={setDitherEndHeight}
      />
      
      {/* Cyberpunk grid background */}
      <div className="absolute inset-0 opacity-10" style={{
        backgroundImage: `linear-gradient(to right, ${themeColors.surface} 1px, transparent 1px), linear-gradient(to bottom, ${themeColors.surface} 1px, transparent 1px)`,
        backgroundSize: '4rem 4rem'
      }} />

      {/* Glow effects - reduced */}
      <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-[120px]" style={{
        backgroundColor: themeColors.secondary,
        opacity: themeMode === "dark" ? 0.05 : 0.02
      }} />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-[120px]" style={{
        backgroundColor: themeColors.primary,
        opacity: themeMode === "dark" ? 0.05 : 0.02
      }} />

      {/* Header */}
      <div className="relative z-50 border-b" style={{ borderColor: `${themeColors.primary}30` }}>
        <div className="container mx-auto px-4 py-4 flex items-center justify-between max-w-5xl">
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r" style={{
              backgroundImage: `linear-gradient(to right, ${themeColors.primary}, ${themeColors.secondary})`
            }}>
              EUGLENA
            </h1>
            <a
              href="https://github.com/muxite/webRAG"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 border text-xs font-mono transition-colors hover:opacity-80"
              style={{
                borderColor: themeColors.primary,
                color: themeColors.primary,
                backgroundColor: themeMode === "dark" ? `${themeColors.surface}E6` : "rgba(255, 255, 255, 0.9)",
                boxShadow: `0 0 8px ${themeColors.primary}20`,
              }}
            >
              <Github className="w-4 h-4" />
              <span>REPO</span>
            </a>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setThemeMode(themeMode === "dark" ? "light" : "dark")}
              className="px-3 py-1.5 border text-xs font-mono transition-colors"
              style={{
                borderColor: themeColors.primary,
                color: themeColors.primary,
                backgroundColor: themeMode === "dark" ? `${themeColors.surface}E6` : "rgba(255, 255, 255, 0.9)",
                boxShadow: `0 0 8px ${themeColors.primary}20`,
              }}
            >
              {themeMode === "dark" ? "L" : "D"}
            </button>
            
            {/* Palette selector */}
            <div className="relative">
              <button
                onClick={() => setShowPaletteMenu(!showPaletteMenu)}
                className="px-3 py-1.5 border text-xs font-mono transition-colors flex items-center gap-1"
                style={{
                  borderColor: themeColors.primary,
                  color: themeColors.primary,
                  backgroundColor: themeMode === "dark" ? `${themeColors.surface}E6` : "rgba(255, 255, 255, 0.9)",
                  boxShadow: `0 0 8px ${themeColors.primary}20`,
                }}
              >
                <Palette className="w-4 h-4" />
              </button>
              
              {showPaletteMenu && (
                <div className="absolute right-0 mt-2 w-48 border z-[100]" style={{
                  backgroundColor: themeColors.surface,
                  borderColor: themeColors.primary,
                  boxShadow: `0 0 20px ${themeColors.primary}30`,
                }}>
                  {colorPalettes.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setPaletteId(p.id);
                        setShowPaletteMenu(false);
                      }}
                      className="w-full px-3 py-2 text-left font-mono text-xs transition-colors hover:opacity-80"
                      style={{
                        backgroundColor: paletteId === p.id ? `${themeColors.primary}20` : "transparent",
                        color: themeColors.text,
                      }}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {showReconnect && (
              <button
                onClick={handleReconnectSupabase}
                className="flex items-center gap-2 px-3 py-1.5 border border-yellow-500 text-yellow-400 hover:bg-yellow-500/10 transition-colors font-mono text-sm"
                style={{
                  backgroundColor: themeMode === "dark" ? `${themeColors.surface}E6` : "rgba(255, 255, 255, 0.9)",
                  boxShadow: "0 0 8px rgba(234, 179, 8, 0.2)",
                }}
              >
                <RefreshCw className="w-4 h-4" />
                Reconnect
              </button>
            )}
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-3 py-1.5 border border-red-500 text-red-400 hover:bg-red-500/10 transition-colors font-mono text-sm"
              style={{
                backgroundColor: themeMode === "dark" ? `${themeColors.surface}E6` : "rgba(255, 255, 255, 0.9)",
                boxShadow: "0 0 8px rgba(239, 68, 68, 0.2)",
              }}
            >
              <LogOut className="w-4 h-4" />
              Log Out
            </button>
          </div>
        </div>
      </div>

      <div className="relative z-10 container mx-auto px-4 sm:px-6 py-4 sm:py-6 space-y-3 sm:space-y-4 max-w-5xl">
        {/* System Info */}
        {systemInfo && (
          <VectorBoxHeavy padding={4} borderColor={themeColors.primary} bgColor={themeColors.boxBgHeavy}>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <h3 className="text-subheading shrink-0">Backend Status</h3>
              <div className="text-green-400 text-status">Online</div>
              <div className="text-metadata" style={{ color: themeColors.text }}>Gateway {systemInfo.gatewayVersion}</div>
              <div className="text-metadata" style={{ color: themeColors.text }}>Worker {systemInfo.workerVersion}</div>
              <div className="text-metadata" style={{ color: themeColors.text }}>{systemInfo.activeWorkers} workers</div>
            </div>
          </VectorBoxHeavy>
        )}

        {/* User Stats */}
        {userStats && (
          <VectorBoxHeavy padding={4} borderColor={themeColors.secondary} bgColor={themeColors.boxBgHeavy}>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <div className="text-metadata" style={{ color: themeColors.text }}>{userStats.email}</div>
              <div className="text-metadata" style={{ color: themeColors.text }}>{userStats.dailyTicks} ticks</div>
            </div>
          </VectorBoxHeavy>
        )}

        {/* Task Submission */}
        <VectorBoxHeavy padding={6} borderColor={themeColors.primary} bgColor={themeColors.boxBgHeavy}>
          <h2 className="text-heading mb-3">New Task</h2>
          <form onSubmit={handleSubmit} className="space-y-3">
            <div>
              <label className="block text-label mb-2">
                Task Description
              </label>
              <textarea
                value={mandate}
                onChange={(e) => setMandate(e.target.value)}
                className="w-full border-2 px-4 py-3 focus:outline-none transition-colors resize-none h-24 text-input"
                style={{
                  backgroundColor: themeColors.surface,
                  borderColor: themeColors.secondary,
                  color: themeColors.text,
                }}
                placeholder="Describe what you want the AI to do..."
                required
              />
            </div>
            <div>
              <label className="block text-label mb-2">
                Max Iterations
              </label>
              <input
                type="number"
                min="1"
                value={maxTicks}
                onChange={(e) => setMaxTicks(parseInt(e.target.value) || 0)}
                className="w-full border-2 px-4 py-2 focus:outline-none transition-colors text-input"
                style={{
                  backgroundColor: themeColors.surface,
                  borderColor: themeColors.secondary,
                  color: themeColors.text,
                }}
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full text-white font-bold py-3 px-6 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              style={{
                background: `linear-gradient(to right, ${themeColors.secondary}, ${themeColors.primary})`,
              }}
            >
              <Send className="w-4 h-4" />
              <span className="text-button">{loading ? "Processing..." : "Submit Task"}</span>
            </button>
            {submitMessage && (
              <div className="text-metadata-secondary text-sm" style={{ color: themeColors.text }}>
                {submitMessage}
              </div>
            )}
          </form>
        </VectorBoxHeavy>

        {/* Task List */}
        <div className="space-y-3">
          <h2 className="text-heading">Task History</h2>
          {tasks.length === 0 ? (
            <VectorBox padding={8} borderColor={themeColors.primary} bgColor={themeColors.boxBg}>
              <div className="text-center py-6 text-metadata-muted">
                No tasks yet
              </div>
            </VectorBox>
          ) : (
            <div className="space-y-3">
              {tasks.map((task, index) => (
                <motion.div
                  key={task.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <TaskContainer task={task} themeColors={themeColors} onDelete={handleDeleteTask} defaultOpen={index === 0} />
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-50 border-t mt-8" style={{ borderColor: `${themeColors.primary}30` }}>
        <div className="container mx-auto px-4 py-6 max-w-5xl">
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