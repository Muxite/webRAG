import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "npm:@supabase/supabase-js";
import * as kv from "./kv_store.tsx";

const app = new Hono();

// Create Supabase client for admin operations
const supabaseAdmin = createClient(
  Deno.env.get("SUPABASE_URL") || "",
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || ""
);

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Health check endpoint
app.get("/make-server-65da8f1f/health", (c) => {
  return c.json({ status: "ok" });
});

// Signup endpoint
app.post("/make-server-65da8f1f/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();

    if (!email || !password) {
      return c.json({ error: "Email and password required" }, 400);
    }

    // Create user with Supabase Auth
    // Automatically confirm the user's email since an email server hasn't been configured.
    const { data, error } = await supabaseAdmin.auth.admin.createUser({
      email,
      password,
      user_metadata: { name: name || "" },
      email_confirm: true,
    });

    if (error) {
      console.error("Signup error:", error);
      return c.json({ error: error.message }, 400);
    }

    // Initialize user data in KV store
    await kv.set(`user:${data.user.id}:ticks`, "1000");
    await kv.set(`user:${data.user.id}:tasks`, JSON.stringify([]));

    return c.json({ success: true, user: data.user });
  } catch (error: any) {
    console.error("Signup error:", error);
    return c.json({ error: error.message || "Signup failed" }, 500);
  }
});

// System info endpoint
app.get("/make-server-65da8f1f/system-info", async (c) => {
  try {
    return c.json({
      title: "CyberLink AI Network",
      backendVersion: "v2.1.47",
      activeWorkers: 42,
      lastUpdate: new Date().toISOString().split("T")[0],
      github: "https://github.com/cyberlink/ai-network",
    });
  } catch (error: any) {
    console.error("System info error:", error);
    return c.json({ error: error.message }, 500);
  }
});

// User stats endpoint (requires authentication)
app.get("/make-server-65da8f1f/user-stats", async (c) => {
  try {
    const accessToken = c.req.header("Authorization")?.split(" ")[1];
    const {
      data: { user },
      error,
    } = await supabaseAdmin.auth.getUser(accessToken || "");

    if (!user || error) {
      return c.json({ error: "Unauthorized" }, 401);
    }

    let ticksStr = await kv.get(`user:${user.id}:ticks`);
    
    // Initialize ticks if not set
    if (!ticksStr) {
      console.log(`Initializing ticks for user ${user.id}`);
      await kv.set(`user:${user.id}:ticks`, "1000");
      ticksStr = "1000";
    }
    
    const ticksRemaining = parseInt(ticksStr || "0");

    return c.json({
      email: user.email,
      ticksRemaining,
      dailyTicks: ticksRemaining,
    });
  } catch (error: any) {
    console.error("User stats error:", error);
    return c.json({ error: error.message }, 500);
  }
});

// Get tasks endpoint (requires authentication)
app.get("/make-server-65da8f1f/tasks", async (c) => {
  try {
    const accessToken = c.req.header("Authorization")?.split(" ")[1];
    const {
      data: { user },
      error,
    } = await supabaseAdmin.auth.getUser(accessToken || "");

    if (!user || error) {
      return c.json({ error: "Unauthorized" }, 401);
    }

    const tasksStr = await kv.get(`user:${user.id}:tasks`);
    const tasks = tasksStr ? JSON.parse(tasksStr) : [];

    return c.json({ tasks });
  } catch (error: any) {
    console.error("Get tasks error:", error);
    return c.json({ error: error.message }, 500);
  }
});

// Submit task endpoint (requires authentication)
app.post("/make-server-65da8f1f/submit-task", async (c) => {
  try {
    const accessToken = c.req.header("Authorization")?.split(" ")[1];
    const {
      data: { user },
      error,
    } = await supabaseAdmin.auth.getUser(accessToken || "");

    if (!user || error) {
      return c.json({ error: "Unauthorized" }, 401);
    }

    const { mandate, maxTicks } = await c.req.json();

    if (!mandate || !maxTicks || maxTicks <= 0) {
      return c.json({ error: "Invalid task parameters" }, 400);
    }

    // Check user has enough ticks
    let ticksStr = await kv.get(`user:${user.id}:ticks`);
    
    // Initialize ticks if not set
    if (!ticksStr) {
      console.log(`Initializing ticks for user ${user.id}`);
      await kv.set(`user:${user.id}:ticks`, "1000");
      ticksStr = "1000";
    }
    
    const ticksRemaining = parseInt(ticksStr || "0");
    
    console.log(`User ${user.id} has ${ticksRemaining} ticks, requesting ${maxTicks}`);

    if (ticksRemaining < maxTicks) {
      return c.json({ 
        error: "Insufficient ticks",
        ticksRemaining,
        ticksRequested: maxTicks 
      }, 400);
    }

    // Create task
    const task = {
      id: crypto.randomUUID(),
      timestamp: new Date().toLocaleString(),
      mandate,
      maxTicks,
      status: "running",
      ticksUsed: Math.floor(Math.random() * maxTicks * 0.8),
      results: "",
    };

    // Add task to user's tasks immediately
    const tasksStr = await kv.get(`user:${user.id}:tasks`);
    const tasks = tasksStr ? JSON.parse(tasksStr) : [];
    const updatedTasks = [task, ...tasks];
    await kv.set(`user:${user.id}:tasks`, JSON.stringify(updatedTasks));

    // Simulate task processing in background
    setTimeout(async () => {
      task.status = "completed";
      task.ticksUsed = Math.floor(Math.random() * maxTicks);
      task.results = `Task completed successfully. Executed ${task.ticksUsed} ticks. AI processed the mandate and generated results based on the directive: "${mandate}". Output: [SIMULATED_AI_RESPONSE_DATA]`;

      const currentTasksStr = await kv.get(`user:${user.id}:tasks`);
      const currentTasks = currentTasksStr ? JSON.parse(currentTasksStr) : [];
      const taskIndex = currentTasks.findIndex((t: any) => t.id === task.id);
      
      if (taskIndex !== -1) {
        currentTasks[taskIndex] = task;
        await kv.set(`user:${user.id}:tasks`, JSON.stringify(currentTasks));
      }

      // Deduct ticks
      const currentTicksStr = await kv.get(`user:${user.id}:ticks`);
      const currentTicks = parseInt(currentTicksStr || "1000");
      const newTicks = currentTicks - task.ticksUsed;
      await kv.set(`user:${user.id}:ticks`, newTicks.toString());
    }, 5000); // Simulate 5 second processing

    return c.json({ success: true, task });
  } catch (error: any) {
    console.error("Submit task error:", error);
    return c.json({ error: error.message }, 500);
  }
});

Deno.serve(app.fetch);