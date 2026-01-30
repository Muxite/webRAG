import React from 'react';
import { theme } from '../../config/theme';

interface AuthSectionProps {
  userEmail: string | null;
  onSignOut: () => void;
}

export function AuthSection({ userEmail, onSignOut }: AuthSectionProps) {
  return (
    <div className={`mb-4 flex items-center justify-between text-sm ${theme.colors.text.primary} font-mono`}>
      <div className="space-y-1">
        <div className="font-bold">Signed in as {userEmail}</div>
        <div className={`text-xs ${theme.colors.text.muted}`}>Daily limit: 32 ticks per day</div>
      </div>
      <button
        onClick={onSignOut}
        className={`px-3 py-1 rounded-lg ${theme.colors.button.danger.bg} ${theme.colors.button.danger.text} font-mono text-sm font-bold active:scale-95 transition-all ${theme.colors.button.danger.hover}`}
      >
        Sign Out
      </button>
    </div>
  );
}
