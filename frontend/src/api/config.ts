import { projectId, publicAnonKey } from "/utils/supabase/info.tsx";
import { supabase } from "@/lib/supabase";

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * VERSION CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * 🔧 EASY API REPLACEMENT:
 * Update these version strings to match your backend gateway and worker versions.
 * These are displayed in the Dashboard's "Backend Status" section.
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 */
export const VERSION_CONFIG = {
  gateway: "v2.2",
  worker: "v3.1",
} as const;

/**
 * Detect if running in local development mode.
 * Uses Vite's dev mode detection or explicit env variable.
 */
const isLocalDev = import.meta.env.DEV || import.meta.env.VITE_USE_LOCAL === "true";

/**
 * Determine the gateway base URL.
 * Priority:
 * 1. Explicit VITE_GATEWAY_URL env variable
 * 2. Local dev mode -> http://localhost:8080 (Docker gateway)
 * 3. Default production URL
 */
const getGatewayBaseUrl = (): string => {
  if (import.meta.env.VITE_GATEWAY_URL) {
    return import.meta.env.VITE_GATEWAY_URL;
  }
  if (isLocalDev) {
    return "http://localhost:8080";
  }
  return "https://euglena-api.com";
};

export const API_CONFIG = {
  gatewayBaseUrl: getGatewayBaseUrl(),
  publicAnonKey: publicAnonKey,
  isLocalDev,
} as const;

if (isLocalDev && !import.meta.env.VITE_GATEWAY_URL) {
  console.log(`🔧 Local Development Mode: Using Docker gateway at ${API_CONFIG.gatewayBaseUrl}`);
  console.log(`   Set VITE_GATEWAY_URL to override or VITE_USE_LOCAL=false to disable`);
}

/**
 * API ENDPOINTS
 */
export const API_ENDPOINTS = {
  systemInfo: `${API_CONFIG.gatewayBaseUrl}/system-info`,
  workerCount: `${API_CONFIG.gatewayBaseUrl}/worker-count`,
  submitTask: `${API_CONFIG.gatewayBaseUrl}/tasks`,
} as const;

export const MOCK_DATA = {
  systemInfo: {
    title: "Euglena Gateway",
    gatewayVersion: VERSION_CONFIG.gateway,
    workerVersion: VERSION_CONFIG.worker,
    activeWorkers: 2,
    lastUpdate: new Date().toISOString().split("T")[0],
    github: "https://github.com/muxite/webRAG",
  },
  userStats: {
    email: "user@example.com",
    ticksRemaining: 3,
    dailyTicks: 3,
  },
  tasks: [
    {
      id: "task-mock-001",
      correlationId: "mock-001",
      timestamp: new Date().toLocaleString(),
      createdAt: new Date().toISOString(),
      completedAt: null,
      mandate: "Draft a short summary of recent system activity.",
      maxTicks: 120,
      status: "in_progress",
      ticksUsed: 32,
      results: "",
      deliverables: "",
      notes: "",
    },
    {
      id: "task-mock-002",
      correlationId: "mock-002",
      timestamp: new Date(Date.now() - 3600_000).toLocaleString(),
      createdAt: new Date(Date.now() - 3600_000).toISOString(),
      completedAt: new Date(Date.now() - 1200_000).toISOString(),
      mandate: "Review queued analysis deliverables and summarize.",
      maxTicks: 80,
      status: "completed",
      ticksUsed: 80,
      results: "",
      deliverables: "Sample deliverable text to show output format.",
      notes: "Sample reasoning notes to show how the result was achieved.",
    },
  ],
} as const;
/**
 * API REQUEST DEFAULTS
 */
export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
} as const;

export const POLLING_INTERVAL = 3000; // milliseconds

/**
 * TYPES
 */
export interface SystemInfo {
  title: string;
  gatewayVersion: string;
  workerVersion: string;
  activeWorkers: number;
  lastUpdate: string;
  github: string;
}

export interface UserStats {
  email: string;
  ticksRemaining: number;
  dailyTicks: number;
}

export interface Task {
  id: string;
  correlationId?: string;
  timestamp: string;
  createdAt?: string;
  completedAt?: string | null;
  mandate: string;
  maxTicks: number;
  status: string;
  ticksUsed: number;
  results: string;
  deliverables?: string;
  notes?: string;
}

export interface TaskSubmission {
  mandate: string;
  maxTicks: number;
}

export interface TaskSubmissionResponse {
  correlation_id: string;
  status: string;
  mandate: string;
  created_at: string;
  updated_at: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
  tick?: number | null;
  max_ticks?: number;
}

export interface SignupData {}

export interface LoginData {
  email: string;
  password: string;
}

/**
 * API SERVICE FUNCTIONS
 */
/**
 * Creates authorization headers with Bearer token
 * 
 * @param token - Optional user JWT token from Supabase auth
 * @returns Headers object with Authorization and Content-Type
 * 
 * 💡 USAGE:
 * - Without token: Uses Supabase anon key (for public endpoints)
 * - With token: Uses user's JWT token (for protected endpoints)
 */
export function createAuthHeaders(token?: string): HeadersInit {
  const headers: HeadersInit = {
    ...DEFAULT_HEADERS,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  } else {
    headers['Authorization'] = `Bearer ${API_CONFIG.publicAnonKey}`;
  }
  
  return headers;
}

/**
 * Fetch system information (backend status, versions, active workers)
 * 
 * 🌐 ENDPOINT: GET /system-info
 * 🔓 AUTH: Public (uses anon key)
 * 
 * @returns SystemInfo object with backend status
 * 
 * 📊 EXPECTED RESPONSE:
 * {
 *   "title": "CyberLink Terminal",
 *   "gatewayVersion": "v1.3.2",
 *   "workerVersion": "v2.1.47",
 *   "activeWorkers": 42,
 *   "lastUpdate": "2026-02-08",
 *   "github": "https://github.com/..."
 * }
 * 
 * 🔧 FALLBACK BEHAVIOR:
 * If the API fails, returns VERSION_CONFIG values for gateway/worker versions.
 * Update VERSION_CONFIG at the top of this file to change displayed versions.
 */
export async function fetchSystemInfo(): Promise<SystemInfo> {
  try {
    const response = await fetch(API_ENDPOINTS.systemInfo, {
      headers: createAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`System info request failed: ${response.status}`);
    }
    const systemInfo = await response.json();
    
    const formatVersion = (version: string | undefined): string => {
      if (!version) return "0.0";
      return version.startsWith("v") ? version : `v${version}`;
    };

    return {
      title: systemInfo.title || "Euglena Gateway",
      gatewayVersion: formatVersion(systemInfo.gatewayVersion),
      workerVersion: formatVersion(systemInfo.workerVersion),
      activeWorkers: systemInfo.activeWorkers ?? 0,
      lastUpdate: systemInfo.lastUpdate || new Date().toISOString().split("T")[0],
      github: systemInfo.github || "https://github.com/muxite/webRAG",
    };
  } catch (error) {
    console.error("Error fetching system info:", error);
    console.warn("Falling back to mock data - gateway may be unavailable");
    return MOCK_DATA.systemInfo;
  }
}

export async function fetchWorkerCount(): Promise<number> {
  try {
    const response = await fetch(API_ENDPOINTS.workerCount, {
      headers: createAuthHeaders(),
    });
    if (!response.ok) {
      throw new Error(`Worker count request failed: ${response.status}`);
    }
    const data = await response.json();
    const activeWorkers = Number(data?.activeWorkers);
    return Number.isFinite(activeWorkers) ? activeWorkers : MOCK_DATA.systemInfo.activeWorkers;
  } catch (error) {
    console.error("Error fetching worker count:", error);
    return MOCK_DATA.systemInfo.activeWorkers;
  }
}

/**
 * Fetch user statistics (email, credit balance, daily allowance)
 * 
 * 🌐 ENDPOINT: GET /user-stats
 * 🔒 AUTH: Protected (requires user token)
 * 
 * @param token - User's JWT token from Supabase
 * @param userEmail - Optional fallback email for mock data
 * @returns UserStats object with credit balance and email
 * 
 * 📊 EXPECTED RESPONSE:
 * {
 *   "email": "user@example.com",
 *   "ticksRemaining": 1000,
 *   "dailyTicks": 250
 * }
 */
export async function fetchUserStats(token: string, userEmail?: string): Promise<UserStats> {
  try {
    const { data: userData, error: userError } = await supabase.auth.getUser(token);
    if (userError || !userData?.user) {
      throw new Error("Unauthorized");
    }

    const user = userData.user;
    const today = new Date().toISOString().split("T")[0];

    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .select("daily_tick_limit")
      .eq("user_id", user.id)
      .maybeSingle();

    if (profileError) {
      throw new Error(`Failed to load profile: ${profileError.message}`);
    }
    
    const { data: usage, error: usageError } = await supabase
      .from("user_daily_usage")
      .select("ticks_used")
      .eq("user_id", user.id)
      .eq("usage_date", today)
      .maybeSingle();

    if (usageError) {
      throw new Error(`Failed to load usage: ${usageError.message}`);
    }

    const dailyLimit = typeof profile?.daily_tick_limit === "number" ? profile.daily_tick_limit : 0;
    const ticksUsed = typeof usage?.ticks_used === "number" ? usage.ticks_used : 0;
    const ticksRemaining = dailyLimit > 0 ? Math.max(dailyLimit - ticksUsed, 0) : 0;

    return {
      email: user.email || userEmail || "",
      ticksRemaining,
      dailyTicks: dailyLimit,
    };
  } catch (error) {
    console.error("Error fetching user stats:", error);
    return {
      ...MOCK_DATA.userStats,
      email: userEmail || MOCK_DATA.userStats.email,
    };
  }
}

/**
 * Fetch all tasks for the current user
 * 
 * 🌐 ENDPOINT: GET /tasks
 * 🔒 AUTH: Protected (requires user token)
 * 
 * @param token - User's JWT token from Supabase
 * @returns Array of Task objects
 * 
 * 📊 EXPECTED RESPONSE:
 * {
 *   "tasks": [
 *     {
 *       "id": "task-001",
 *       "timestamp": "2026-02-08T10:30:00Z",
 *       "mandate": "Analyze market trends",
 *       "maxTicks": 500,
 *       "status": "completed",
 *       "ticksUsed": 342,
 *       "results": "Analysis complete..."
 *     }
 *   ]
 * }
 */
export async function fetchTasks(token: string): Promise<Task[]> {
  try {
    const { data, error } = await supabase
      .from("tasks")
      .select("id, correlation_id, mandate, status, max_ticks, tick, result, error, created_at, updated_at")
      .order("created_at", { ascending: false });

    if (error) {
      throw new Error(`Failed to load tasks: ${error.message}`);
    }

    return (data || []).map((task) => {
      const createdAt = task.created_at || "";
      const completedAt = task.status === "completed" ? task.updated_at || null : null;
      const { deliverables, notes, results } = formatTaskResultParts(task.result, task.error, completedAt);

      return {
        id: task.id || task.correlation_id || "",
        correlationId: task.correlation_id || undefined,
        timestamp: createdAt ? new Date(createdAt).toLocaleString() : "",
        createdAt,
        completedAt,
        mandate: task.mandate || "",
        maxTicks: task.max_ticks || 0,
        status: task.status || "unknown",
        ticksUsed: task.tick || 0,
        results,
        deliverables,
        notes,
      };
    });
  } catch (error) {
    console.error("Error fetching tasks:", error);
    return [...MOCK_DATA.tasks];
  }
}

/**
 * Submit a new task to the backend
 * 
 * 🌐 ENDPOINT: POST /submit-task
 * 🔒 AUTH: Protected (requires user token)
 * 
 * @param token - User's JWT token from Supabase
 * @param taskData - Task submission data (mandate, maxTicks)
 * @returns Boolean indicating success/failure
 * 
 * 📤 REQUEST BODY:
 * {
 *   "mandate": "Write a comprehensive report",
 *   "maxTicks": 500
 * }
 * 
 * 📊 EXPECTED RESPONSE (Success):
 * {
 *   "id": "task-003",
 *   "timestamp": "2026-02-08T12:00:00Z",
 *   "mandate": "Write a comprehensive report",
 *   "maxTicks": 500,
 *   "status": "pending",
 *   "ticksUsed": 0,
 *   "results": ""
 * }
 */
export async function submitTask(
  token: string,
  taskData: TaskSubmission
): Promise<TaskSubmissionResponse | null> {
  try {
    const response = await fetch(API_ENDPOINTS.submitTask, {
      method: 'POST',
      headers: createAuthHeaders(token),
      body: JSON.stringify({ mandate: taskData.mandate, max_ticks: taskData.maxTicks }),
    });

    if (!response.ok) {
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('Error submitting task:', error);
    return null;
  }
}

export async function deleteTask(token: string, taskId: string): Promise<boolean> {
  try {
    if (!taskId) {
      return false;
    }

    const { error } = await supabase
      .from("tasks")
      .delete()
      .eq("id", taskId);

    if (error) {
      throw new Error(error.message);
    }

    return true;
  } catch (error) {
    console.error("Error deleting task:", error);
    return false;
  }
}

/**
 * Split task result into deliverables, notes, and fallback text.
 *
 * @param result - Raw result object from Supabase.
 * @param error - Error string from Supabase.
 * @param completedAt - Completion timestamp for fallback text.
 * @returns Parsed deliverables, notes, and fallback results text.
 */
function formatTaskResultParts(
  result: unknown,
  error: string | null,
  completedAt: string | null
): { deliverables?: string; notes?: string; results: string } {
  if (error) {
    return { results: error };
  }

  let deliverablesText = "";
  let notesText = "";
  let resultsText = "";

  if (result && typeof result === "object") {
    const data = result as Record<string, unknown>;
    const deliverables = data.deliverables;
    if (Array.isArray(deliverables)) {
      deliverablesText = deliverables.map((item) => String(item)).join("\n");
    } else if (deliverables != null) {
      deliverablesText = String(deliverables);
    }

    const notes =
      data.notes ??
      data.action_summary ??
      data.thought_process ??
      data.analysis ??
      data.reasoning;
    if (notes != null) {
      notesText = String(notes);
    }
  }

  if (!deliverablesText && !notesText && result != null) {
    resultsText = typeof result === "string" ? result : JSON.stringify(result);
  }

  if (!deliverablesText && !notesText && completedAt) {
    const completedLine = `Completed: ${new Date(completedAt).toLocaleString()}`;
    resultsText = resultsText ? `${completedLine}\n${resultsText}` : completedLine;
  }

  return {
    deliverables: deliverablesText || undefined,
    notes: notesText || undefined,
    results: resultsText,
  };
}

export async function signupUser(_: SignupData): Promise<{ success: boolean; error?: string }> {
  return { success: false, error: "Direct signup endpoint removed; use Supabase Auth instead." };
}