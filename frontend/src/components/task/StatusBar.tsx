import React from 'react';
import { TaskResponse } from '../../services/api';
import { theme } from '../../config/theme';
import { Loader2, Clock, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { Progress } from '../ui/progress';

interface StatusBarProps {
  task: TaskResponse | null;
  status?: string;
}

export function StatusBar({ task, status }: StatusBarProps) {
  if (!task) {
    return null;
  }

  const currentStatus = status || task.status?.toLowerCase() || 'unknown';
  const hasTicks = task.tick !== undefined && task.max_ticks > 0;
  const progressPercent = hasTicks ? Math.min(100, (task.tick / task.max_ticks) * 100) : 0;

  const getStatusConfig = () => {
    switch (currentStatus) {
      case 'in_queue':
      case 'pending':
        return {
          icon: Clock,
          iconColor: theme.colors.status.inProgress.icon,
          bgColor: theme.colors.status.inProgress.bg,
          borderColor: theme.colors.status.inProgress.border,
          textColor: theme.colors.status.inProgress.text,
          label: 'In Queue',
          progressColor: 'bg-blue-500',
        };
      case 'in_progress':
      case 'accepted':
        return {
          icon: Loader2,
          iconColor: theme.colors.status.inProgress.icon,
          bgColor: theme.colors.status.inProgress.bg,
          borderColor: theme.colors.status.inProgress.border,
          textColor: theme.colors.status.inProgress.text,
          label: 'In Progress',
          progressColor: 'bg-blue-500',
        };
      case 'completed':
        return {
          icon: CheckCircle2,
          iconColor: theme.colors.status.completed.icon,
          bgColor: theme.colors.status.completed.bg,
          borderColor: theme.colors.status.completed.border,
          textColor: theme.colors.status.completed.text,
          label: 'Completed',
          progressColor: 'bg-green-500',
        };
      case 'error':
      case 'failed':
        return {
          icon: XCircle,
          iconColor: theme.colors.status.error.icon,
          bgColor: theme.colors.status.error.bg,
          borderColor: theme.colors.status.error.border,
          textColor: theme.colors.status.error.text,
          label: 'Error',
          progressColor: 'bg-red-500',
        };
      default:
        return {
          icon: AlertCircle,
          iconColor: theme.colors.status.pending.icon,
          bgColor: theme.colors.status.pending.bg,
          borderColor: theme.colors.status.pending.border,
          textColor: theme.colors.status.pending.text,
          label: currentStatus,
          progressColor: 'bg-yellow-500',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;
  const isSpinning = currentStatus === 'in_progress' || currentStatus === 'accepted';

  return (
    <div className={`p-5 rounded-xl ${config.bgColor} border-2 ${config.borderColor}`}>
      <div className="flex items-center gap-3 mb-3">
        <Icon
          size={24}
          className={`${config.iconColor} ${isSpinning ? 'animate-spin' : ''}`}
        />
        <div className="flex-1">
          <p className={`font-mono font-bold ${config.textColor} text-base`}>
            {config.label}
          </p>
          {hasTicks && (
            <p className={`font-mono text-sm ${theme.colors.text.secondary} opacity-75 mt-1`}>
              Tick {task.tick} of {task.max_ticks}
            </p>
          )}
        </div>
        {hasTicks && (
          <div className="text-right">
            <p className={`font-mono text-base font-bold ${config.textColor}`}>
              {Math.round(progressPercent)}%
            </p>
          </div>
        )}
      </div>
      {hasTicks && (
        <div className="mt-3">
          <Progress value={progressPercent} className="h-2.5" />
        </div>
      )}
      {task.correlation_id && (
        <p className={`font-mono text-sm ${theme.colors.text.tertiary} opacity-60 mt-3`}>
          ID: {task.correlation_id.slice(0, 8)}...
        </p>
      )}
    </div>
  );
}
