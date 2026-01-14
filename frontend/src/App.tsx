import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiClient, TaskResponse } from './services/api';
import { supabase } from './services/supabaseClient';
import { getApiMode, setApiMode, getCurrentApiBaseURL } from './config/api';
import { Header } from './components/layout/Header';
import { ApiModeToggle } from './components/layout/ApiModeToggle';
import { AboutSection } from './components/layout/AboutSection';
import { LoginForm } from './components/auth/LoginForm';
import { AuthSection } from './components/auth/AuthSection';
import { TaskForm } from './components/task/TaskForm';
import { TaskStatusDisplay } from './components/task/TaskStatusDisplay';

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
  const [workerCount, setWorkerCount] = useState<number | null>(null);
  const [rotationAngle, setRotationAngle] = useState(0);
  const animationFrameRef = useRef<number | null>(null);
  const lastApiCallAngleRef = useRef<number>(-1);
  
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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const pollTaskStatus = useCallback(async (correlationId: string) => {
    try {
      const updatedTask = await apiClient.getTask(correlationId);
      setTask(updatedTask);

      if (updatedTask.status === 'completed' || updatedTask.status === 'error' || updatedTask.status === 'failed') {
        setLoading(false);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch task status';
      if (errorMessage.includes('404') || errorMessage.includes('not found')) {
        setError('Task not found');
        setLoading(false);
      } else {
        console.error('Error polling task status:', err);
      }
    }
  }, []);

  useEffect(() => {
    const pollWorkerCount = async () => {
      try {
        const response = await apiClient.getWorkerCount();
        setWorkerCount(response.count);
      } catch (err) {
        console.error('Failed to fetch worker count:', err);
      }
    };

    pollWorkerCount();

    const animate = () => {
      setRotationAngle((prev) => {
        const newAngle = (prev + 2) % 360;
        
        const prevAngle = prev;
        const crossedTop = prevAngle >= 358 && newAngle < 2;
        
        if (crossedTop) {
          lastApiCallAngleRef.current = newAngle;
          pollWorkerCount();
          
          if (task && task.status !== 'completed' && task.status !== 'error' && task.status !== 'failed') {
            pollTaskStatus(task.correlation_id);
          }
        }
        
        return newAngle;
      });
      
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [task, pollTaskStatus]);

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
    setTask(null);

    try {
      const response = await apiClient.submitTask({
        mandate: mandate.trim(),
        max_ticks: maxTicks,
      });

      setTask(response);
      setLoading(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to submit task';
      setError(errorMessage);
      setLoading(false);
    }
  };

  const handleReset = () => {
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

  const handleApiModeChange = () => {
    const nextMode = apiMode === 'localhost' ? 'aws' : apiMode === 'aws' ? 'auto' : 'localhost';
    setApiMode(nextMode);
    setApiModeState(nextMode);
    const newUrl = getCurrentApiBaseURL();
    setCurrentApiUrl(newUrl);
    apiClient.updateBaseURL(newUrl);
  };

  const bg = dark ? 'bg-zinc-900' : 'bg-white';
  const text = dark ? 'text-white' : 'text-zinc-900';
  const isAuthenticated = !!userEmail;
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
        <Header text={text} workerCount={workerCount} />

        <div className="space-y-8 p-12 rounded-3xl" style={randomColor}>
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
                hasTask={!!task}
              />

              <TaskStatusDisplay
                task={task}
                error={error}
                rotationAngle={rotationAngle}
                isOutOfTicks={isOutOfTicks}
                remainingTicks={remainingTicks}
              />
            </>
          )}
        </div>

        <AboutSection dark={dark} text={text} />
      </div>
    </div>
  );
}
