import React from 'react';
import { Server } from 'lucide-react';
import { theme } from '../../config/theme';

interface ApiModeToggleProps {
  apiMode: 'localhost' | 'aws' | 'auto';
  currentApiUrl: string;
  text: string;
  isLocalhost: boolean;
  onToggle: () => void;
}

export function ApiModeToggle({
  apiMode,
  currentApiUrl,
  text,
  isLocalhost,
  onToggle,
}: ApiModeToggleProps) {
  if (!isLocalhost) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 flex gap-2 z-50">
      <div className="relative group">
        <button
          onClick={onToggle}
          className={`p-2 hover:opacity-70 active:scale-95 transition-all rounded-lg border-2 ${theme.colors.background.secondary}`}
          style={{
            borderColor: apiMode === 'localhost' ? '#10b981' : apiMode === 'aws' ? '#3b82f6' : '#6b7280',
            backgroundColor: apiMode === 'localhost' ? 'rgba(16, 185, 129, 0.1)' : apiMode === 'aws' ? 'rgba(59, 130, 246, 0.1)' : 'rgb(39, 39, 42)',
          }}
          title={`API: ${apiMode === 'localhost' ? 'Localhost' : apiMode === 'aws' ? 'AWS' : 'Auto'} (${currentApiUrl})`}
        >
          <Server size={20} className={theme.colors.text.primary} />
        </button>
        <div className={`absolute right-0 top-full mt-2 p-2 rounded-lg text-xs font-mono ${theme.colors.background.secondary} ${theme.colors.text.primary} ${theme.colors.border.primary} border shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap`}>
          <div className="font-bold">API: {apiMode === 'localhost' ? 'Localhost' : apiMode === 'aws' ? 'AWS' : 'Auto'}</div>
          <div className={`${theme.colors.text.muted} mt-1`}>{currentApiUrl}</div>
          <div className={`${theme.colors.text.placeholder} mt-1`}>Click to switch</div>
        </div>
      </div>
    </div>
  );
}
