import React, { useState, useEffect, useRef } from 'react';
import { Sun, Moon, Github, ExternalLink, Server } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient, TaskResponse } from './services/api';
import { supabase } from './services/supabaseClient';
import { getApiMode, setApiMode, getCurrentApiBaseURL, API_CONFIG } from './config/api';

export default function App() {
  const [mandate, setMandate] = useState('');
  const [maxTicks, setMaxTicks] = useState(32);
  const [maxTicksInput, setMaxTicksInput] = useState('32');
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [error, setError] = useState('');
  const [authError, setAuthError] = useState('');
  const [loading, setLoading] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [dark, setDark] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [apiMode, setApiModeState] = useState<'localhost' | 'aws' | 'auto'>(getApiMode());
  const [currentApiUrl, setCurrentApiUrl] = useState(getCurrentApiBaseURL());
  const pollIntervalRef = useRef<number | null>(null);
  const batchPollIntervalRef = useRef<number | null>(null);
  const batchTasksRef = useRef<Array<{ correlationId: string; status: string; result?: any }>>([]);
  const [batchTest, setBatchTest] = useState<{
    loading: boolean;
    tasks: Array<{ correlationId: string; status: string; result?: any }>;
    startTime?: number;
    endTime?: number;
    completed: number;
  }>({
    loading: false,
    tasks: [],
    completed: 0,
  });
  
  const isLocalhost = typeof window !== 'undefined' && (
    window.location.hostname === 'localhost' || 
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname.startsWith('192.168.') ||
    window.location.hostname.startsWith('10.')
  );

  const colors = [
    { background: 'linear-gradient(135deg, #4c1d95 0%, #581c87 100%)' },
    { background: 'linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)' },
    { background: 'linear-gradient(135deg, #312e81 0%, #3730a3 100%)' },
  ];
  const [randomColor] = useState(colors[Math.floor(Math.random() * colors.length)]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setUserEmail(data.session?.user.email ?? null);
      if (window.location.hash) {
        window.history.replaceState(null, '', window.location.pathname);
      }
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUserEmail(session?.user.email ?? null);
      if (window.location.hash && session) {
        window.history.replaceState(null, '', window.location.pathname);
      }
    });

    setCurrentApiUrl(getCurrentApiBaseURL());
    setApiModeState(getApiMode());

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      if (batchPollIntervalRef.current) {
        clearInterval(batchPollIntervalRef.current);
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

  const handleDebugQueueTest = async () => {
    if (!userEmail) {
      setError('You must be logged in to run debug queue test');
      return;
    }

    setError('');

    const debugPhrase = 'debugdebugdebug';
    try {
      const response = await apiClient.submitTask({
        mandate: `${debugPhrase} test message`,
        max_ticks: 1,
      });
      setError(`Debug queue test: Published 1 message to debug queue (correlation_id: ${response.correlation_id})`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to publish to debug queue';
      if (errorMessage.includes('429') || errorMessage.includes('Too Many Requests')) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        try {
          const response = await apiClient.submitTask({
            mandate: `${debugPhrase} test message`,
            max_ticks: 1,
          });
          setError(`Debug queue test: Published 1 message to debug queue (correlation_id: ${response.correlation_id})`);
        } catch (retryErr) {
          setError(`Debug queue test failed: ${retryErr instanceof Error ? retryErr.message : 'Failed to submit after retry'}`);
        }
      } else {
        setError(`Debug queue test failed: ${errorMessage}`);
      }
    }
  };

  const handleBatchSkipTest = async () => {
    if (!userEmail) {
      setError('You must be logged in to run skip test');
      return;
    }

    setError('');

    try {
      const skipPhrase = 'skipskipskip';
      const response = await apiClient.submitTask({
        mandate: `${skipPhrase} test`,
        max_ticks: 1,
      });
      setError(`Skip test: Published 1 message (correlation_id: ${response.correlation_id})`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to submit';
      if (errorMessage.includes('429') || errorMessage.includes('Too Many Requests')) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        try {
          const response = await apiClient.submitTask({
            mandate: `${skipPhrase} test`,
            max_ticks: 1,
          });
          setError(`Skip test: Published 1 message (correlation_id: ${response.correlation_id})`);
        } catch (retryErr) {
          setError(`Skip test failed: ${retryErr instanceof Error ? retryErr.message : 'Failed to submit after retry'}`);
        }
      } else {
        setError(`Skip test failed: ${errorMessage}`);
      }
    }
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
    try {
      const { error: signOutError } = await supabase.auth.signOut();
      if (signOutError) {
        setAuthError(signOutError.message || 'Failed to sign out');
      }
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Failed to sign out. Please check your connection.');
    }
  };

  const bg = dark ? 'bg-zinc-900' : 'bg-white';
  const text = dark ? 'text-white' : 'text-zinc-900';
  const inputBg = 'bg-white text-black';
  const isAuthenticated = !!userEmail;

  const handleApiModeChange = (mode: 'localhost' | 'aws' | 'auto') => {
    setApiMode(mode);
    setApiModeState(mode);
  };

  return (
    <div className={`min-h-screen ${bg} ${text} p-4`}>
      <div className="fixed top-4 right-4 flex gap-2 z-50">
        {isLocalhost && (
          <div className="relative group">
            <button
              onClick={() => {
                const nextMode = apiMode === 'localhost' ? 'aws' : apiMode === 'aws' ? 'auto' : 'localhost';
                handleApiModeChange(nextMode);
              }}
              className="p-2 hover:opacity-70 active:scale-95 transition-all rounded-lg border-2"
              style={{
                borderColor: apiMode === 'localhost' ? '#10b981' : apiMode === 'aws' ? '#3b82f6' : '#6b7280',
                backgroundColor: apiMode === 'localhost' ? 'rgba(16, 185, 129, 0.1)' : apiMode === 'aws' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
              }}
              title={`API: ${apiMode === 'localhost' ? 'Localhost' : apiMode === 'aws' ? 'AWS' : 'Auto'} (${currentApiUrl})`}
            >
              <Server size={20} className={text} />
            </button>
            <div className="absolute right-0 top-full mt-2 p-2 rounded-lg text-xs font-mono bg-zinc-800 text-white border border-zinc-600 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap">
              <div>API: {apiMode === 'localhost' ? 'Localhost' : apiMode === 'aws' ? 'AWS' : 'Auto'}</div>
              <div className="text-zinc-400 mt-1">{currentApiUrl}</div>
              <div className="text-zinc-500 mt-1">Click to switch</div>
            </div>
          </div>
        )}
        <button
          onClick={() => setDark(!dark)}
          className="p-2 hover:opacity-70 active:scale-95 transition-all"
        >
          {dark ? <Sun size={20} className={text} /> : <Moon size={20} className={text} />}
        </button>
      </div>

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
          <div className="flex gap-3">
            <a
              href="https://github.com/Muxite/webRAG"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:opacity-70 transition-opacity"
            >
              <Github size={24} className={text} />
            </a>
            <a
              href="https://muksite.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:opacity-70 transition-opacity"
            >
              <ExternalLink size={24} className={text} />
            </a>
          </div>
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
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={maxTicksInput}
                      onChange={(e) => {
                        const value = e.target.value;
                        if (value === '' || /^\d+$/.test(value)) {
                          setMaxTicksInput(value);
                          if (value !== '') {
                            const numValue = Number(value);
                            if (!isNaN(numValue)) {
                              const clampedValue = Math.max(1, Math.min(32, numValue));
                              setMaxTicks(clampedValue);
                            }
                          }
                        }
                      }}
                      onBlur={(e) => {
                        const numValue = Number(e.target.value);
                        if (isNaN(numValue) || numValue < 1) {
                          setMaxTicks(1);
                          setMaxTicksInput('1');
                        } else if (numValue > 32) {
                          setMaxTicks(32);
                          setMaxTicksInput('32');
                        } else {
                          setMaxTicks(numValue);
                          setMaxTicksInput(String(numValue));
                        }
                      }}
                      disabled={loading}
                      className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono disabled:opacity-50`}
                    />
                    <p className="font-mono text-xs mt-1" style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
                      Each user is restricted to 32 ticks per day across all tasks.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex gap-4 flex-wrap">
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
                <button
                  onClick={handleBatchSkipTest}
                  disabled={loading || !userEmail}
                  className="px-4 py-2 bg-yellow-500 text-black rounded-xl border-2 border-yellow-600 disabled:opacity-50
                  font-mono active:scale-95 transition-all hover:bg-yellow-400 hover:border-yellow-500"
                >
                  Skip Test
                </button>
                <button
                  onClick={handleDebugQueueTest}
                  disabled={loading || !userEmail}
                  className="px-4 py-2 bg-orange-500 text-white rounded-xl border-2 border-orange-600 disabled:opacity-50
                  font-mono active:scale-95 transition-all hover:bg-orange-400 hover:border-orange-500"
                >
                  Debug Test
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

              {batchTest.tasks.length > 0 && (
                <div className="p-6 rounded-xl bg-yellow-50 border-2 border-yellow-200">
                  <h3 className="font-mono font-bold mb-4 text-yellow-900 uppercase tracking-wide">
                    Batch Skip Test Results
                  </h3>
                  {batchTest.startTime && batchTest.endTime && (
                    <div className="mb-4 p-3 bg-white rounded-lg border border-yellow-300">
                      <p className="font-mono text-sm">
                        <span className="font-bold">Total Time:</span>{' '}
                        {((batchTest.endTime - batchTest.startTime) / 1000).toFixed(2)}s
                      </p>
                      <p className="font-mono text-sm">
                        <span className="font-bold">Average Time per Task:</span>{' '}
                        {((batchTest.endTime - batchTest.startTime) / (32 * 1000)).toFixed(2)}s
                      </p>
                    </div>
                  )}
                  <div className="mb-4">
                    <div className="mb-2 flex items-center justify-between text-sm font-mono">
                      <span className="font-bold">Progress</span>
                      <span>{batchTest.completed} / 32 completed</span>
                    </div>
                    <div className="h-6 bg-gray-200 rounded-full overflow-hidden relative">
                      <div
                        className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2"
                        style={{
                          width: `${(batchTest.completed / 32) * 100}%`,
                          background: 'linear-gradient(to right, #fbbf24, #f59e0b)',
                        }}
                      >
                        <span className="text-xs font-bold text-black">
                          {Math.round((batchTest.completed / 32) * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {batchTest.tasks.map((task, index) => (
                      <div
                        key={task.correlationId}
                        className="p-2 rounded bg-white border border-yellow-200 text-xs font-mono"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-bold">Task {index + 1}:</span>
                          <span
                            className={
                              task.status === 'completed'
                                ? 'text-green-600'
                                : task.status === 'error' || task.status === 'failed'
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }
                          >
                            {task.status}
                          </span>
                        </div>
                        {task.result?.error && (
                          <p className="text-red-600 mt-1">{task.result.error}</p>
                        )}
                      </div>
                    ))}
                  </div>
                  {batchTest.completed === 32 && batchTest.endTime && (
                    <div className="mt-4 p-3 bg-green-100 rounded-lg border border-green-300">
                      <p className="font-mono text-sm text-green-800 font-bold">
                        All 32 tasks completed successfully!
                      </p>
                    </div>
                  )}
                </div>
              )}

              {task && (
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-white text-black">
                    {task.tick !== undefined && task.max_ticks > 0 && (
                      <div className="mb-4">
                        <div className="mb-2 flex items-center justify-between text-sm font-mono">
                          <span className="font-bold">Usage</span>
                          <span>{task.tick} / {task.max_ticks} ticks</span>
                        </div>
                        <div className="h-6 bg-gray-200 rounded-full overflow-hidden relative">
                          <div
                            className={`h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2 ${
                              task.status !== 'completed' && task.status !== 'error' ? 'animate-pulse' : ''
                            }`}
                            style={{
                              width: `${Math.min(100, (task.tick / task.max_ticks) * 100)}%`,
                              background: 'linear-gradient(to right, #3b82f6, #a855f7)',
                            }}
                          >
                            <span className="text-xs font-bold text-white">
                              {Math.round((task.tick / task.max_ticks) * 100)}%
                            </span>
                          </div>
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
            <div className="pt-4">
              <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">How the Scraper Works</h3>
              <p className="text-xs opacity-80 mb-2">
                When the agent needs to gather information from the web, it follows a two-step process:
              </p>
              <ul className="space-y-1 text-xs opacity-80">
                <li>
                  • <span className="font-bold">Web Search:</span> Uses the Brave Search API to find relevant URLs based on the task mandate. The search returns titles, URLs, and descriptions that help the agent identify which pages to visit.
                </li>
                <li>
                  • <span className="font-bold">Content Extraction:</span> When visiting a URL, the agent makes an HTTP GET request to fetch the HTML content. It then uses BeautifulSoup to parse the HTML, removing script and style tags, and extracts the main text content. This cleaned text is stored in the agent's observations and can be retrieved from ChromaDB for future reference.
                </li>
                <li>
                  • <span className="font-bold">Memory Storage:</span> Extracted content is embedded and stored in ChromaDB, allowing the agent to recall relevant information across multiple ticks and tasks through semantic similarity search.
                </li>
              </ul>
            </div>
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
            <div className="pt-4">
              <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Attributions</h3>
              <ul className="space-y-1 text-xs opacity-80">
                <li>
                  • UI components from <a href="https://ui.shadcn.com/" target="_blank" rel="noopener noreferrer" className="underline">shadcn/ui</a> used under <a href="https://github.com/shadcn-ui/ui/blob/main/LICENSE.md" target="_blank" rel="noopener noreferrer" className="underline">MIT license</a>
                </li>
                <li>
                  • Icons from <a href="https://lucide.dev/" target="_blank" rel="noopener noreferrer" className="underline">Lucide</a> used under <a href="https://github.com/lucide-icons/lucide/blob/main/LICENSE" target="_blank" rel="noopener noreferrer" className="underline">ISC license</a>
                </li>
                <li>
                  • Web scraping powered by <a href="https://www.crummy.com/software/BeautifulSoup/" target="_blank" rel="noopener noreferrer" className="underline">BeautifulSoup</a>
                </li>
                <li>
                  • Search functionality via <a href="https://brave.com/search/api/" target="_blank" rel="noopener noreferrer" className="underline">Brave Search API</a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
