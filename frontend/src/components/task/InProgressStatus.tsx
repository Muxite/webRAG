import React from 'react';
import { TaskResponse } from '../../services/api';
import { theme } from '../../config/theme';

interface InProgressStatusProps {
  task: TaskResponse;
}

export function InProgressStatus({ task }: InProgressStatusProps) {
  return (
    <div className={`p-5 rounded-xl ${theme.colors.background.card} ${theme.colors.text.primary}`}>
      <div className="space-y-3 font-mono text-base">
        <p className={theme.colors.text.primary}>
          <span className={`font-bold ${theme.colors.text.primary}`}>Status:</span> <span className={theme.colors.text.primary}>{task.status || 'in_progress'}</span>
        </p>
        {task.correlation_id && (
          <p className={`text-sm ${theme.colors.text.secondary} opacity-75`}>
            <span className={`font-bold ${theme.colors.text.secondary}`}>ID:</span> <span className={theme.colors.text.secondary}>{task.correlation_id.slice(0, 8)}...</span>
          </p>
        )}
      </div>
    </div>
  );
}
