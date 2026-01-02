import { useState, useEffect, useRef } from 'react';
import { Sun, Moon, Github } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
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

  const colors = [
    { background: 'linear-gradient(135deg, #4c1d95 0%, #581c87 100%)' }, // purple-900
    { background: 'linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)' }, // blue-900
    { background: 'linear-gradient(135deg, #312e81 0%, #3730a3 100%)' }, // indigo-900
  ];
  const [randomColor] = useState(colors[Math.floor(Math.random() * colors.length)]);

  useEffect(() => {
    // Handle email confirmation redirect
    supabase.auth.getSession().then(({ data }) => {
      setUserEmail(data.session?.user.email ?? null);
      // Clear URL hash after processing
      if (window.location.hash) {
        window.history.replaceState(null, '', window.location.pathname);
      }
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUserEmail(session?.user.email ?? null);
      // Clear URL hash after auth state change (e.g., after email confirmation)
      if (window.location.hash && session) {
        window.history.replaceState(null, '', window.location.pathname);
      }
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
        className="fixed top-4 right-4 p-2 hover:opacity-70 active:scale-95 transition-all z-50"
      >
        {dark ? <Sun size={20} className={text} /> : <Moon size={20} className={text} />}
      </button>

      <div className="w-full max-w-4xl mx-auto">
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h1
              className="uppercase tracking-wider"
              style={{
                fontFamily: 'Impact, Arial Black, sans-serif',
                fontSize: '4.5rem',
                background: 'linear-gradient(to right, #ff00ff, #60a5fa, #38bdf8)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              EUGLENA
            </h1>
            <p className={`${text} mt-2 font-mono text-sm`}>
              Autonomous RAG agent for web automation and task execution
            </p>
          </div>
          <a
            href="https://github.com/Muxite/webRAG"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 hover:opacity-70 transition-opacity"
          >
            <Github size={24} className={text} />
          </a>
        </div>

        <div className="space-y-8 p-12 rounded-3xl" style={randomColor}>
          {!isAuthenticated ? (
            <>
              <h2
                className="uppercase tracking-wider"
                style={{
                  fontFamily: 'Impact, Arial Black, sans-serif',
                  fontSize: '2rem',
                  background: 'linear-gradient(to right, #ff00ff, #60a5fa, #38bdf8)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Sign In
              </h2>
              <div className="max-w-md space-y-4">
                <p className="text-white font-mono text-sm">
                  Create an account or sign in to submit tasks to Euglena.
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
                className="uppercase tracking-wider"
                style={{
                  fontFamily: 'Impact, Arial Black, sans-serif',
                  fontSize: '2rem',
                  background: 'linear-gradient(to right, #ff00ff, #60a5fa, #38bdf8)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                TASK SUBMISSION
              </h2>

              <div className="mb-4 flex items-center justify-between text-xs text-white font-mono">
                <div className="space-y-1">
                  <div>Signed in as {userEmail}</div>
                  <div className="opacity-80">Daily limit: 32 ticks per day</div>
                </div>
                <button
                  onClick={handleSignOut}
                  className="px-3 py-1 rounded-lg text-white font-mono active:scale-95 transition-all"
                  style={{
                    backgroundColor: 'rgb(220, 38, 38)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgb(239, 68, 68)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgb(220, 38, 38)';
                  }}
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
                    <label className="block text-white font-mono mb-2">
                      Max ticks
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={32}
                      value={maxTicks}
                      onChange={(e) => setMaxTicks(Number(e.target.value) || 1)}
                      disabled={loading}
                      className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono disabled:opacity-50`}
                    />
                    <p className="font-mono text-xs mt-1" style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                      Each user is restricted to 32 ticks per day across all tasks.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !mandate.trim() || !userEmail}
                  className="px-4 py-2 bg-white text-black rounded-xl border-2 border-gray-200 disabled:opacity-50
                  font-mono active:scale-95 transition-all
                  disabled:hover:bg-white disabled:hover:border-gray-200 disabled:active:scale-100"
                  onMouseEnter={(e) => {
                    if (!e.currentTarget.disabled) {
                      e.currentTarget.style.backgroundColor = '#a855f7';
                      e.currentTarget.style.color = 'white';
                      e.currentTarget.style.borderColor = '#9333ea';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!e.currentTarget.disabled) {
                      e.currentTarget.style.backgroundColor = 'white';
                      e.currentTarget.style.color = 'black';
                      e.currentTarget.style.borderColor = '#e5e7eb';
                    }
                  }}
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
                    {task.tick !== undefined && task.max_ticks > 0 && (
                      <div className="mb-4">
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${Math.min(100, (task.tick / task.max_ticks) * 100)}%`,
                              background: 'linear-gradient(to right, #3b82f6, #a855f7)',
                            }}
                          />
                        </div>
                      </div>
                    )}
                    <div 
                      className="h-1 rounded-full mb-4"
                      style={{
                        backgroundColor: task.status === 'completed' ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
                      }}
                    />
                    <div className="space-y-2 font-mono">
                      <p>
                        <span className="font-bold">Status:</span> {task.status}
                      </p>
                      {task.tick !== undefined && (
                        <p>
                          <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
                        </p>
                      )}
                      {task.result?.success !== undefined && (
                        <p>
                          <span className="font-bold">Success:</span>{' '}
                          <span className={task.result.success ? 'text-green-600' : 'text-red-600'}>
                            {task.result.success ? 'Yes' : 'No'}
                          </span>
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
                    <div className="space-y-4">
                      {task.result.deliverables && task.result.deliverables.length > 0 && (
                        <div className="p-6 rounded-xl bg-green-50 border-2 border-green-200">
                          <h3 className="font-mono font-bold mb-4 text-green-900 uppercase tracking-wide">
                            Deliverables
                          </h3>
                          <div className="space-y-4 text-gray-800">
                            {task.result.deliverables.map((deliverable, index) => (
                              <div key={index} className="markdown-content">
                                <ReactMarkdown
                                  components={{
                                    p: ({ children }) => <p className="mb-2 font-mono">{children}</p>,
                                    h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-4 font-mono">{children}</h1>,
                                    h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 font-mono">{children}</h2>,
                                    h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-2 font-mono">{children}</h3>,
                                    ul: ({ children }) => <ul className="font-mono" style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ul>,
                                    ol: ({ children }) => <ol className="font-mono" style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ol>,
                                    li: ({ children }) => <li className="font-mono" style={{ marginBottom: '0.25rem' }}>{children}</li>,
                                    code: ({ children }) => <code className="bg-gray-200 px-1 rounded font-mono text-xs">{children}</code>,
                                    pre: ({ children }) => <pre className="bg-gray-200 p-2 rounded overflow-x-auto mb-2 font-mono text-xs">{children}</pre>,
                                    strong: ({ children }) => <strong className="font-bold font-mono">{children}</strong>,
                                    em: ({ children }) => <em className="italic font-mono">{children}</em>,
                                    a: ({ href, children }) => <a href={href} className="text-blue-600 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
                                  }}
                                >
                                  {deliverable}
                                </ReactMarkdown>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {task.result.notes && (
                        <div className="p-6 rounded-xl bg-blue-50 border-2 border-blue-200">
                          <h3 className="font-mono font-bold mb-4 text-blue-900 uppercase tracking-wide">
                            Notes
                          </h3>
                          <div className="text-gray-800 markdown-content">
                            <ReactMarkdown
                              components={{
                                p: ({ children }) => <p className="mb-2 font-mono">{children}</p>,
                                h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-4 font-mono">{children}</h1>,
                                h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 font-mono">{children}</h2>,
                                h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-2 font-mono">{children}</h3>,
                                ul: ({ children }) => <ul className="font-mono" style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ul>,
                                ol: ({ children }) => <ol className="font-mono" style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ol>,
                                li: ({ children }) => <li className="font-mono" style={{ marginBottom: '0.25rem' }}>{children}</li>,
                                code: ({ children }) => <code className="bg-gray-200 px-1 rounded font-mono text-xs">{children}</code>,
                                pre: ({ children }) => <pre className="bg-gray-200 p-2 rounded overflow-x-auto mb-2 font-mono text-xs">{children}</pre>,
                                strong: ({ children }) => <strong className="font-bold font-mono">{children}</strong>,
                                em: ({ children }) => <em className="italic font-mono">{children}</em>,
                                a: ({ href, children }) => <a href={href} className="text-blue-600 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
                              }}
                            >
                              {task.result.notes}
                            </ReactMarkdown>
                          </div>
                        </div>
                      )}
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
              Euglena is an autonomous RAG agent service that executes tasks through iterative reasoning, web interaction, and vector database storage. Tasks flow through a Gateway API, are consumed by Agent Workers via RabbitMQ, with status tracked in Redis and memory persisted in ChromaDB.
            </p>
            <p>
              The agent uses LLM-powered reasoning to break down tasks, perform web searches, visit URLs, and build up knowledge over time through persistent memory.
            </p>
            <p className="opacity-80">
              <span className="font-bold">Usage Limits:</span> Each user is restricted to 32 ticks per day. This limit applies across all tasks submitted in a 24-hour period.
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
