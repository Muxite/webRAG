import { useState, useEffect, useRef } from 'react';
import { Sun, Moon, Github } from 'lucide-react';
import { apiClient, TaskResponse } from './services/api';
import { supabase } from './services/supabaseClient';

export default function App() {
  const [mandate, setMandate] = useState('');
  const [maxTicks, setMaxTicks] = useState(32);
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [error, setError] = useState('');
  const [authError, setAuthError] = useState('');
  const [loading, setLoading] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [dark, setDark] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  const colors = ['bg-purple-600', 'bg-blue-600', 'bg-indigo-600'];
  const [randomColor] = useState(colors[Math.floor(Math.random() * colors.length)]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setUserEmail(data.session?.user.email ?? null);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUserEmail(session?.user.email ?? null);
    });

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  const pollTaskStatus = async (correlationId: string) => {
    try {
      const updatedTask = await apiClient.getTask(correlationId);
      setTask(updatedTask);

      if (updatedTask.status === 'completed' || updatedTask.status === 'error') {
        stopPolling();
        setLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch task status');
      stopPolling();
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!mandate.trim()) {
      setError('Please enter a mandate');
      return;
    }

    if (!userEmail) {
      setError('You must be logged in to submit a task');
      return;
    }

    setLoading(true);
    setError('');
    setTask(null);

    try {
      const response = await apiClient.submitTask({
        mandate: mandate.trim(),
        max_ticks: maxTicks,
      });

      setTask(response);

      if (response.status !== 'completed' && response.status !== 'error') {
        pollIntervalRef.current = window.setInterval(
          () => pollTaskStatus(response.correlation_id),
          1000
        );
      } else {
        setLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit task');
      setLoading(false);
    }
  };

  const handleReset = () => {
    stopPolling();
    setTask(null);
    setError('');
    setMandate('');
    setLoading(false);
  };

  const handleSignUp = async () => {
    setAuthError('');
    setAuthLoading(true);
    try {
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
      });
      if (signUpError) {
        if ((signUpError as any).code === 'user_already_exists') {
          setAuthError('An account with this email already exists. Please log in instead.');
        } else {
          setAuthError(signUpError.message);
        }
      } else {
        if (data?.session) {
          setAuthError('');
        } else {
          setAuthError('Check your email to confirm your account.');
        }
      }
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Failed to sign up');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignIn = async () => {
    setAuthError('');
    setAuthLoading(true);
    try {
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (signInError) {
        setAuthError(signInError.message);
      }
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Failed to sign in');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleSignOut = async () => {
    setAuthError('');
    setLoading(false);
    stopPolling();
    setTask(null);
    await supabase.auth.signOut();
  };

  const bg = dark ? 'bg-zinc-900' : 'bg-white';
  const text = dark ? 'text-white' : 'text-zinc-900';
  const inputBg = 'bg-white text-black';
  const isAuthenticated = !!userEmail;

  return (
    <div className={`min-h-screen ${bg} ${text} p-4`}>
      <button
        onClick={() => setDark(!dark)}
        className="fixed top-6 right-6 p-4 rounded-xl bg-white text-black border-4 border-gray-200
         hover:bg-yellow-300 hover:border-yellow-400 active:scale-95 transition-all z-50"
      >
        {dark ? <Sun size={24} /> : <Moon size={24} />}
      </button>

      <div className="w-full max-w-6xl mx-auto mb-8 pt-8">
        <div className="flex items-center justify-between">
          <div>
            <h1
              className={`uppercase tracking-wider ${text}`}
              style={{ fontFamily: 'Impact, Arial Black, sans-serif', fontSize: '3rem' }}
            >
              EUGLENA WEB AGENT
            </h1>
            <p className={`${text} mt-2 font-mono`}>Autonomous RAG agent system for web automation</p>
          </div>
          <a
            href="https://github.com/Muxite/webRAG"
            target="_blank"
            rel="noopener noreferrer"
            className={`p-3 rounded-lg ${
              dark
                ? 'bg-zinc-800 hover:bg-zinc-700 border-2 border-zinc-600'
                : 'bg-zinc-300 hover:bg-zinc-400 border-2 border-zinc-400'
            } transition-colors`}
          >
            <Github size={24} />
          </a>
        </div>
      </div>

      <div className="w-full max-w-4xl mx-auto">
        <div className={`space-y-8 p-12 rounded-3xl ${randomColor}`}>
          {!isAuthenticated ? (
            <>
              <h2
                className="uppercase tracking-wider text-white"
                style={{ fontFamily: 'Impact, Arial Black, sans-serif', fontSize: '2rem' }}
              >
                Sign In
              </h2>
              <div className="max-w-md space-y-4">
                <p className="text-white font-mono text-sm">
                  Create an account or sign in to submit tasks to the Euglena agent.
                </p>
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono`}
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono`}
                />
                <div className="flex gap-3">
                  <button
                    onClick={handleSignUp}
                    disabled={authLoading || !email || !password}
                    className="flex-1 p-3 bg-white text-black rounded-xl border-4 border-gray-200 disabled:opacity-50
                    font-mono hover:bg-green-400 hover:border-green-500 active:scale-95 transition-all"
                  >
                    Sign Up
                  </button>
                  <button
                    onClick={handleSignIn}
                    disabled={authLoading || !email || !password}
                    className="flex-1 p-3 bg-black text-white rounded-xl border-4 border-gray-800 disabled:opacity-50
                     hover:bg-blue-500 hover:border-blue-400 active:scale-95 transition-all"
                  >
                    Log In
                  </button>
                </div>
                {authError && (
                  <p className="text-xs text-red-200 font-mono">{authError}</p>
                )}
              </div>
            </>
          ) : (
            <>
              <h2
                className="uppercase tracking-wider text-white"
                style={{ fontFamily: 'Impact, Arial Black, sans-serif', fontSize: '2rem' }}
              >
                TASK SUBMISSION
              </h2>

              <div className="mb-4 flex items-center justify-between text-xs text-white font-mono">
                <span>Signed in as {userEmail}</span>
                <button
                  onClick={handleSignOut}
                  className="px-3 py-1 rounded-lg bg-red-600 border-2 border-red-400 text-white font-mono hover:bg-red-500 active:scale-95 transition-all"
                >
                  Sign Out
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-white font-mono text-sm uppercase tracking-wide">
                    Task Settings
                  </h3>
                  <div>
                    <label className="block text-white font-mono mb-2">Mandate</label>
                    <textarea
                      placeholder="Enter your task mandate..."
                      value={mandate}
                      onChange={(e) => setMandate(e.target.value)}
                      disabled={loading}
                      className={`w-full p-4 rounded-xl ${inputBg} placeholder:text-gray-400 h-32 font-mono disabled:opacity-50`}
                    />
                  </div>
                  <div>
                    <label className="block text-white font-mono mb-2">Max ticks</label>
                    <input
                      type="number"
                      min={1}
                      max={256}
                      value={maxTicks}
                      onChange={(e) => setMaxTicks(Number(e.target.value) || 1)}
                      disabled={loading}
                      className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono disabled:opacity-50`}
                    />
                  </div>
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !mandate.trim() || !userEmail}
                  className="flex-1 p-6 bg-white text-black rounded-xl border-4 border-gray-200 disabled:opacity-50
                  font-mono hover:bg-green-400 hover:border-green-500 active:scale-95 transition-all
                  disabled:hover:bg-white disabled:hover:border-gray-200 disabled:active:scale-100"
                >
                  {loading ? 'Processing...' : 'Submit Task'}
                </button>
                {task && (
                  <button
                    onClick={handleReset}
                    disabled={loading}
                    className="px-6 py-4 bg-black text-white rounded-xl border-4 border-gray-800 font-mono
                    hover:bg-red-600 hover:border-red-400 active:scale-95 transition-all disabled:opacity-50"
                  >
                    Reset
                  </button>
                )}
              </div>

              {error && (
                <div className="p-6 rounded-xl bg-red-100 text-red-800 border-2 border-red-300">
                  <p className="font-mono font-bold">Error: {error}</p>
                </div>
              )}

              {task && (
                <div className="space-y-4">
                  <div className="p-6 rounded-xl bg-white text-black">
                    <div className="space-y-2 font-mono">
                      <p>
                        <span className="font-bold">Status:</span> {task.status}
                      </p>
                      <p>
                        <span className="font-bold">Correlation ID:</span> {task.correlation_id}
                      </p>
                      {task.tick !== undefined && (
                        <p>
                          <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
                        </p>
                      )}
                      {task.error && (
                        <p className="text-red-600">
                          <span className="font-bold">Error:</span> {task.error}
                        </p>
                      )}
                    </div>
                  </div>

                  {task.result && (
                    <div className="p-6 rounded-xl bg-white text-black">
                      <h3 className="font-mono font-bold mb-2">Result:</h3>
                      <pre className="whitespace-pre-wrap font-mono text-sm">
                        {JSON.stringify(task.result, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <div className={`mt-8 p-8 rounded-2xl ${
          dark ? 'bg-zinc-800 border-2 border-zinc-600' : 'bg-zinc-200 border-2 border-zinc-400'
        }`}>
          <h2
            className="uppercase tracking-wide text-lg mb-4"
            style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}
          >
            About Euglena
          </h2>
          <div className="space-y-4 font-mono text-sm">
            <p>
              A distributed agent framework where tasks flow through a Gateway API, are consumed by
              Agent Workers via RabbitMQ, with status tracked in Redis and memory persisted in
              ChromaDB.
            </p>
            <p>
              The agent uses LLM-powered reasoning to break down tasks, perform web searches, visit
              URLs, and build up knowledge over time.
            </p>
            <div className="pt-4">
              <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Architecture</h3>
              <ul className="space-y-1 text-xs opacity-80">
                <li>
                  • <span className="font-bold">Gateway:</span> FastAPI service accepting tasks via
                  REST API
                </li>
                <li>
                  • <span className="font-bold">Agent Workers:</span> Consume tasks from RabbitMQ
                  and execute them
                </li>
                <li>
                  • <span className="font-bold">Status Tracking:</span> Real-time monitoring in
                  Redis
                </li>
                <li>
                  • <span className="font-bold">Memory:</span> Context retention in ChromaDB vector
                  database
                </li>
              </ul>
            </div>
            <div className="pt-4">
              <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Tech Stack</h3>
              <p className="text-xs opacity-80">
                Built with FastAPI, RabbitMQ, Redis, ChromaDB, and Docker. Powered by OpenAI and
                web search APIs.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
