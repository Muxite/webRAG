import React, { useState } from 'react';
import { TaskResponse } from '../../services/api';
import { theme } from '../../config/theme';
import { StatusBar } from './StatusBar';
import { InProgressStatus } from './InProgressStatus';
import { CompletedStatus } from './CompletedStatus';
import { ErrorStatus } from './ErrorStatus';

interface TaskCardProps {
  task: TaskResponse;
}

export function TaskCard({ task }: TaskCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const status = task.status?.toLowerCase() || 'unknown';

  const formatDate = (dateString: string): { date: string; time: string } => {
    try {
      const date = new Date(dateString);
      const dateStr = date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
      const timeStr = date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      });
      return { date: dateStr, time: timeStr };
    } catch {
      return { date: dateString, time: '' };
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'error':
      case 'failed':
        return 'bg-red-500';
      case 'in_progress':
      case 'in_queue':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  const progressPercent = task.tick && task.max_ticks > 0
    ? Math.min(100, (task.tick / task.max_ticks) * 100)
    : 0;

  const { date, time } = formatDate(task.created_at);

  const getStatusBadgeClass = (status: string): string => {
    switch (status) {
      case 'completed':
        return theme.colors.status.completed.badge;
      case 'error':
      case 'failed':
        return theme.colors.status.error.badge;
      case 'in_progress':
      case 'in_queue':
        return theme.colors.status.inProgress.badge;
      default:
        return theme.colors.status.default.badge;
    }
  };

  return (
    <div className={`${theme.colors.background.card} rounded-xl ${theme.colors.border.primary} border-2 overflow-hidden`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full p-2 text-left ${theme.colors.text.primary} hover:bg-zinc-700 transition-colors flex items-center gap-2`}
      >
        <span className={`text-xs font-mono ${theme.colors.text.primary} whitespace-nowrap flex-shrink-0`}>
          {date}
        </span>
        <span className={`text-xs font-mono ${theme.colors.text.primary} whitespace-nowrap flex-shrink-0`}>
          {time}
        </span>
        <span className={`text-sm font-mono font-bold px-2 py-0.5 rounded flex-shrink-0 ${getStatusBadgeClass(status)}`}>
          {status.toUpperCase()}
        </span>
        {(task.tick !== undefined && task.max_ticks > 0) && (
          <>
            <div className={`w-16 ${theme.colors.background.tertiary} rounded-full h-1.5 flex-shrink-0`}>
              <div
                className={`h-1.5 rounded-full ${getStatusColor(status)}`}
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <span className={`text-xs font-mono ${theme.colors.text.primary} whitespace-nowrap flex-shrink-0`}>
              {task.tick}/{task.max_ticks}
            </span>
          </>
        )}
        <p className={`text-xs font-mono ${theme.colors.text.primary} break-words flex-1 min-w-0 text-left`}>
          {task.mandate}
        </p>
        <div className="flex-shrink-0">
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className={`border-t ${theme.colors.border.primary} p-4 space-y-4 ${theme.colors.background.primary}`}>
          <div className="space-y-2">
            <h4 className={`text-sm font-mono font-bold ${theme.colors.text.primary}`}>Full Mandate</h4>
            <div className={`${theme.colors.background.card} p-3 rounded-lg border ${theme.colors.border.primary}`}>
              <p className={`text-sm ${theme.colors.text.primary} font-mono whitespace-pre-wrap break-words leading-relaxed`}>
                {task.mandate}
              </p>
            </div>
          </div>

          <StatusBar task={task} status={status} />

          {status === 'in_progress' || status === 'in_queue' || status === 'accepted' ? (
            <InProgressStatus task={task} />
          ) : status === 'completed' ? (
            <CompletedStatus task={task} />
          ) : status === 'error' || status === 'failed' ? (
            <ErrorStatus task={task} />
          ) : null}

          {task.result && (
            <div className="space-y-2">
              <h4 className={`text-sm font-mono font-bold ${theme.colors.text.primary}`}>Result</h4>
              <div className={`${theme.colors.background.card} p-3 rounded-lg border ${theme.colors.border.primary} space-y-3`}>
                {task.result.deliverables && task.result.deliverables.length > 0 && (
                  <div>
                    <h5 className={`text-xs font-mono font-bold ${theme.colors.text.primary} mb-1`}>Deliverables:</h5>
                    <ul className="list-disc list-inside space-y-1">
                      {task.result.deliverables.map((deliverable, idx) => (
                        <li key={idx} className={`text-sm ${theme.colors.text.primary} font-mono leading-relaxed`}>
                          {deliverable}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {task.result.notes && (
                  <div>
                    <h5 className={`text-xs font-mono font-bold ${theme.colors.text.primary} mb-1`}>Notes:</h5>
                    <p className={`text-sm ${theme.colors.text.primary} font-mono whitespace-pre-wrap break-words leading-relaxed`}>
                      {task.result.notes}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {task.error && (
            <div className="space-y-1">
              <h4 className={`text-sm font-mono font-bold ${theme.colors.status.error.icon}`}>Error</h4>
              <div className={`${theme.colors.status.error.bg} p-3 rounded-lg border ${theme.colors.status.error.border}`}>
                <p className={`text-sm ${theme.colors.status.error.text} font-mono break-words leading-relaxed`}>
                  {task.error}
                </p>
              </div>
            </div>
          )}

          <div className={`pt-3 border-t ${theme.colors.border.primary}`}>
            <div className={`text-xs ${theme.colors.text.primary} font-mono space-y-0.5`}>
              <div>Created: {date} {time}</div>
              <div>Updated: {formatDate(task.updated_at).date} {formatDate(task.updated_at).time}</div>
              <div>ID: {task.correlation_id}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
