import React from 'react';

interface AuthSectionProps {
  userEmail: string | null;
  onSignOut: () => void;
}

export function AuthSection({ userEmail, onSignOut }: AuthSectionProps) {
  return (
    <div className="mb-4 flex items-center justify-between text-xs text-white font-mono">
      <div className="space-y-1">
        <div>Signed in as {userEmail}</div>
        <div className="opacity-80">Daily limit: 32 ticks per day</div>
      </div>
      <button
        onClick={onSignOut}
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
  );
}
