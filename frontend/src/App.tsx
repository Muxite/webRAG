import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { apiClient, TaskResponse } from './services/api';
import { supabase } from './services/supabaseClient';
import { getApiMode, setApiMode, getCurrentApiBaseURL } from './config/api';
import { theme } from './config/theme';
import { Header } from './components/layout/Header';
import { ApiModeToggle } from './components/layout/ApiModeToggle';
import { AboutSection } from './components/layout/AboutSection';
import { LoginForm } from './components/auth/LoginForm';
import { AuthSection } from './components/auth/AuthSection';
import { TaskForm } from './components/task/TaskForm';
import { TaskCard } from './components/task/TaskCard';

export default function App() {
  const [mandate, setMandate] = useState('');
  const [maxTicks, setMaxTicks] = useState(32);
  const [maxTicksInput, setMaxTicksInput] = useState('32');
  const [tasks, setTasks] = useState<TaskResponse[]>([]);
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
  const [workerCount, setWorkerCount] = useState<number | null>(null);
  const [gatewayConnected, setGatewayConnected] = useState(false);
  const [gatewayConnecting, setGatewayConnecting] = useState(false);
  
  const isAuthenticated = useMemo(() => !!userEmail, [userEmail]);
  
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
      if (!session) {
        setTasks([]);
        setGatewayConnected(false);
      }
    });

    setCurrentApiUrl(getCurrentApiBaseURL());
    setApiModeState(getApiMode());

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  const loadTaskHistory = useCallback(async () => {
    try {
      const taskList = await apiClient.listTasks();
      setTasks(taskList);
    } catch (err) {
      console.error('Failed to load task history:', err);
    }
  }, []);

  const pollTaskStatus = useCallback(async (correlationId: string) => {
    try {
      const updatedTask = await apiClient.getTask(correlationId);
      setTasks(prevTasks => {
        const index = prevTasks.findIndex(t => t.correlation_id === correlationId);
        if (index >= 0) {
          const newTasks = [...prevTasks];
          newTasks[index] = updatedTask;
          return newTasks;
        }
        return [updatedTask, ...prevTasks];
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch task status';
      if (!errorMessage.includes('404') && !errorMessage.includes('not found')) {
        console.error('Error polling task status:', err);
      }
    }
  }, []);

  const waitForGatewayConnection = useCallback(async (maxRetries: number = 10, initialDelay: number = 1000): Promise<boolean> => {
    setGatewayConnecting(true);
    let delay = initialDelay;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        await apiClient.checkHealth();
        setGatewayConnected(true);
        setGatewayConnecting(false);
        return true;
      } catch (err) {
        if (attempt < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay));
          delay = Math.min(delay * 1.5, 5000);
        }
      }
    }
    
    setGatewayConnecting(false);
    return false;
  }, []);

  useEffect(() => {
    if (!isAuthenticated) {
      setGatewayConnected(false);
      return;
    }

    const initializeGateway = async () => {
      const connected = await waitForGatewayConnection();
      if (connected) {
        await loadTaskHistory();
      }
    };

    initializeGateway();
  }, [isAuthenticated, waitForGatewayConnection, loadTaskHistory]);

  useEffect(() => {
    if (!gatewayConnected || !isAuthenticated) {
      return;
    }

    const pollWorkerCount = async () => {
      try {
        const response = await apiClient.getWorkerCount();
        setWorkerCount(response.count);
      } catch (err) {
        console.error('Failed to fetch worker count:', err);
      }
    };

    pollWorkerCount();

    const interval = setInterval(() => {
      pollWorkerCount();
      tasks.forEach(task => {
        if (task.status !== 'completed' && task.status !== 'error' && task.status !== 'failed') {
          pollTaskStatus(task.correlation_id);
        }
      });
    }, 5000);

    return () => {
      clearInterval(interval);
    };
  }, [gatewayConnected, isAuthenticated, tasks, pollTaskStatus]);

  const parseOutOfTicksError = (errorMessage: string): { isOutOfTicks: boolean; remaining: number } => {
    if (errorMessage.includes('Daily tick limit exceeded')) {
      const match = errorMessage.match(/Remaining ticks: (\d+)/);
      const remaining = match ? parseInt(match[1], 10) : 0;
      return { isOutOfTicks: true, remaining };
    }
    return { isOutOfTicks: false, remaining: 0 };
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

    try {
      const response = await apiClient.submitTask({
        mandate: mandate.trim(),
        max_ticks: maxTicks,
      });

      setTasks(prevTasks => [response, ...prevTasks]);
      setMandate('');
      setLoading(false);
      await loadTaskHistory();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to submit task';
      setError(errorMessage);
      setLoading(false);
    }
  };

  const handleReset = () => {
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
    setTasks([]);
    try {
      const { error: signOutError } = await supabase.auth.signOut();
      if (signOutError) {
        setAuthError(signOutError.message || 'Failed to sign out');
      }
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Failed to sign out. Please check your connection.');
    }
  };

  const handleMaxTicksChange = (value: string) => {
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
  };

  const handleMaxTicksBlur = () => {
    const numValue = Number(maxTicksInput);
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
  };

  const handleApiModeChange = async () => {
    const nextMode = apiMode === 'localhost' ? 'aws' : apiMode === 'aws' ? 'auto' : 'localhost';
    setApiMode(nextMode);
    setApiModeState(nextMode);
    const newUrl = getCurrentApiBaseURL();
    setCurrentApiUrl(newUrl);
    apiClient.updateBaseURL(newUrl);
    
    setGatewayConnected(false);
    if (isAuthenticated) {
      const connected = await waitForGatewayConnection();
      if (connected) {
        await loadTaskHistory();
      }
    }
  };

  const bg = theme.colors.background.primary;
  const text = theme.colors.text.primary;
  const { isOutOfTicks, remaining: remainingTicks } = parseOutOfTicksError(error);

  return (
    <div className={`min-h-screen ${bg} ${text} p-4`}>
      <ApiModeToggle
        apiMode={apiMode}
        currentApiUrl={currentApiUrl}
        text={text}
        isLocalhost={isLocalhost}
        onToggle={handleApiModeChange}
      />

      <div className="w-full max-w-4xl mx-auto">
        <div className={`mb-4 p-3 rounded-lg ${theme.colors.status.pending.bg} ${theme.colors.status.pending.border} border-2`}>
          <p className={`${theme.colors.status.pending.text} font-mono text-sm font-bold`}>
            Note: This is an interim version. A frontend overhaul is coming soon.
          </p>
        </div>
        {isAuthenticated && gatewayConnecting && (
          <div className={`mb-4 p-3 rounded-lg ${theme.colors.status.pending.bg} ${theme.colors.status.pending.border} border-2`}>
            <p className={`${theme.colors.status.pending.text} font-mono text-sm`}>
              Connecting to gateway at {currentApiUrl}...
            </p>
          </div>
        )}
        {isAuthenticated && !gatewayConnected && !gatewayConnecting && (
          <div className={`mb-4 p-3 rounded-lg ${theme.colors.status.error.bg} ${theme.colors.status.error.border} border-2`}>
            <p className={`${theme.colors.status.error.text} font-mono text-sm`}>
              Cannot connect to gateway at {currentApiUrl}. Please ensure the gateway service is running.
            </p>
          </div>
        )}
        <Header text={text} workerCount={workerCount} />

        <div className={`space-y-8 p-12 rounded-3xl ${theme.colors.background.secondary} ${theme.colors.border.primary} border-2`}>
          {!isAuthenticated ? (
            <LoginForm
              email={email}
              password={password}
              authError={authError}
              authLoading={authLoading}
              onEmailChange={setEmail}
              onPasswordChange={setPassword}
              onSignUp={handleSignUp}
              onSignIn={handleSignIn}
            />
          ) : (
            <>
              <AuthSection userEmail={userEmail} onSignOut={handleSignOut} />
              <TaskForm
                mandate={mandate}
                maxTicks={maxTicks}
                maxTicksInput={maxTicksInput}
                loading={loading}
                onMandateChange={setMandate}
                onMaxTicksChange={handleMaxTicksChange}
                onMaxTicksBlur={handleMaxTicksBlur}
                onSubmit={handleSubmit}
                onReset={handleReset}
              />

              {error && !isOutOfTicks && (
                <div className={`p-4 rounded-xl ${theme.colors.status.error.bg} ${theme.colors.status.error.text} ${theme.colors.status.error.border} border-2`}>
                  <p className="font-mono font-bold">Error: {error}</p>
                </div>
              )}

              {isOutOfTicks && (
                <div className={`p-4 rounded-xl ${theme.colors.status.pending.bg} ${theme.colors.status.pending.text} ${theme.colors.status.pending.border} border-2`}>
                  <p className="font-mono font-bold">
                    Daily tick limit exceeded. Remaining ticks: {remainingTicks}
                  </p>
                </div>
              )}

              {tasks.length > 0 && (
                <div className="space-y-3">
                  <h3 className={`${theme.colors.text.primary} font-mono text-base font-bold uppercase tracking-wide`}>
                    Task History ({tasks.length})
                  </h3>
                  <div className="space-y-3 max-h-[600px] overflow-y-auto">
                    {tasks.map((task) => (
                      <TaskCard key={task.correlation_id} task={task} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <AboutSection text={text} />
      </div>
    </div>
  );
}
