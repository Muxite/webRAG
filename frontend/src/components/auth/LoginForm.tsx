import React from 'react';
import { theme } from '../../config/theme';

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
  const inputBg = `${theme.colors.background.input} ${theme.colors.text.primary} ${theme.colors.border.input}`;

  return (
    <>
      <h2 className={`${theme.colors.text.primary} font-mono text-xl font-bold uppercase tracking-wide mb-4`}>
        Sign In
      </h2>
      <div className="max-w-md space-y-4">
        <p className={`${theme.colors.text.primary} font-mono text-sm`}>
          Create an account or sign in to submit tasks to Euglena.
        </p>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => onEmailChange(e.target.value)}
          className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-500 font-mono text-sm border`}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => onPasswordChange(e.target.value)}
          className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-500 font-mono text-sm border`}
        />
        <div className="flex gap-3">
          <button
            onClick={onSignUp}
            disabled={authLoading || !email || !password}
            className={`flex-1 p-3 ${theme.colors.button.secondary.bg} ${theme.colors.button.secondary.text} rounded-xl border-2 ${theme.colors.button.secondary.border} disabled:opacity-50
            font-mono text-sm font-bold hover:bg-zinc-600 hover:border-zinc-500 active:scale-95 transition-all`}
          >
            Sign Up
          </button>
          <button
            onClick={onSignIn}
            disabled={authLoading || !email || !password}
            className={`flex-1 p-3 ${theme.colors.button.primary.bg} ${theme.colors.button.primary.text} rounded-xl border-2 ${theme.colors.button.primary.border} disabled:opacity-50
            font-mono text-sm font-bold hover:bg-gray-200 hover:border-gray-400 active:scale-95 transition-all`}
          >
            Log In
          </button>
        </div>
        {authError && (
          <p className={`text-xs ${theme.colors.status.error.text} font-mono font-bold`}>{authError}</p>
        )}
      </div>
    </>
  );
}
