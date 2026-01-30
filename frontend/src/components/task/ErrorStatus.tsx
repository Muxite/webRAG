import React from 'react';
import { XCircle } from 'lucide-react';
import { TaskResponse } from '../../services/api';
import { theme } from '../../config/theme';

interface ErrorStatusProps {
  task: TaskResponse;
}

export function ErrorStatus({ task }: ErrorStatusProps) {
  const progressPercent = task.tick !== undefined && task.max_ticks > 0
    ? Math.min(100, (task.tick / task.max_ticks) * 100)
    : 0;

  return (
    <div className="space-y-4">
      <div className={`p-5 rounded-xl ${theme.colors.background.card} ${theme.colors.text.primary}`}>
        {task.tick !== undefined && task.max_ticks > 0 && (
          <div className="mb-5">
            <div className={`mb-3 flex items-center justify-between text-base font-mono`}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Usage</span>
              <span className={theme.colors.text.primary}>{task.tick} / {task.max_ticks} ticks</span>
            </div>
            <div className={`h-7 ${theme.colors.background.tertiary} rounded-full overflow-hidden relative`}>
              <div
                className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-3"
                style={{
                  width: `${progressPercent}%`,
                  background: 'linear-gradient(to right, #ef4444, #dc2626)',
                }}
              >
                <span className="text-sm font-bold text-white">
                  {Math.round(progressPercent)}%
                </span>
              </div>
            </div>
          </div>
        )}
        <div className="h-1.5 rounded-full mb-5 bg-red-500" />
        <div className="space-y-3 font-mono text-base">
          <div className="flex items-center gap-3">
            <XCircle size={24} className={theme.colors.status.error.icon} />
            <p className={theme.colors.text.primary}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Status:</span> <span className={theme.colors.text.primary}>{task.status}</span>
            </p>
          </div>
          {task.tick !== undefined && (
            <p className={theme.colors.text.primary}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Tick:</span> <span className={theme.colors.text.primary}>{task.tick} / {task.max_ticks}</span>
            </p>
          )}
          {task.error && (
            <div className={`p-4 ${theme.colors.status.error.bg} border ${theme.colors.status.error.border} rounded-lg`}>
              <p className={`text-base ${theme.colors.status.error.text} break-words leading-relaxed`}>
                <span className="font-bold">Error:</span> {task.error}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
