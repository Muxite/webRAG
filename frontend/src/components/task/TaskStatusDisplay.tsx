import React from 'react';
import { TaskResponse } from '../../services/api';
import { NoResponseStatus } from './NoResponseStatus';
import { OutOfTicksStatus } from './OutOfTicksStatus';
import { InProgressStatus } from './InProgressStatus';
import { CompletedStatus } from './CompletedStatus';
import { ErrorStatus } from './ErrorStatus';
import { StatusBar } from './StatusBar';

interface TaskStatusDisplayProps {
  task: TaskResponse | null;
  error: string;
  isOutOfTicks: boolean;
  remainingTicks?: number;
}

export function TaskStatusDisplay({
  task,
  error,
  isOutOfTicks,
  remainingTicks = 0,
}: TaskStatusDisplayProps) {
  if (isOutOfTicks) {
    return <OutOfTicksStatus remaining={remainingTicks} error={error} />;
  }

  if (error && !task) {
    const isNetworkError = error.includes('Cannot connect') || error.includes('Network error') || error.includes('Failed to fetch');
    if (isNetworkError) {
      return <NoResponseStatus error={error} />;
    }
    return (
      <div className="p-6 rounded-xl bg-red-100 text-red-800 border-2 border-red-300">
        <p className="font-mono font-bold">Error: {error}</p>
      </div>
    );
  }

  if (!task) {
    return null;
  }

  const status = task.status?.toLowerCase() || 'unknown';
  
  return (
    <div className="space-y-4">
      <StatusBar task={task} status={status} />
      
      {status === 'in_progress' || status === 'accepted' ? (
        <InProgressStatus task={task} />
      ) : status === 'completed' ? (
        <CompletedStatus task={task} />
      ) : status === 'error' || status === 'failed' ? (
        <ErrorStatus task={task} />
      ) : null}
    </div>
  );
}
