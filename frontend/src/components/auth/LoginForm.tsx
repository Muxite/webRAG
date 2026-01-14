import React from 'react';

interface LoginFormProps {
  email: string;
  password: string;
  authError: string;
  authLoading: boolean;
  onEmailChange: (email: string) => void;
  onPasswordChange: (password: string) => void;
  onSignUp: () => void;
  onSignIn: () => void;
}

export function LoginForm({
  email,
  password,
  authError,
  authLoading,
  onEmailChange,
  onPasswordChange,
  onSignUp,
  onSignIn,
}: LoginFormProps) {
  const inputBg = 'bg-white text-black';

  return (
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
          onChange={(e) => onEmailChange(e.target.value)}
          className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono`}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => onPasswordChange(e.target.value)}
          className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono`}
        />
        <div className="flex gap-3">
          <button
            onClick={onSignUp}
            disabled={authLoading || !email || !password}
            className="flex-1 p-3 bg-white text-black rounded-xl border-4 border-gray-200 disabled:opacity-50
            font-mono hover:bg-green-400 hover:border-green-500 active:scale-95 transition-all"
          >
            Sign Up
          </button>
          <button
            onClick={onSignIn}
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
  );
}
