import React from 'react';
import { TaskResponse } from '../../services/api';
import { NoResponseStatus } from './NoResponseStatus';
import { OutOfTicksStatus } from './OutOfTicksStatus';
import { InQueueStatus } from './InQueueStatus';
import { InProgressStatus } from './InProgressStatus';
import { CompletedStatus } from './CompletedStatus';
import { ErrorStatus } from './ErrorStatus';
import { UnknownStatus } from './UnknownStatus';

interface TaskStatusDisplayProps {
  task: TaskResponse | null;
  error: string;
  rotationAngle: number;
  isOutOfTicks: boolean;
  remainingTicks?: number;
}

export function TaskStatusDisplay({
  task,
  error,
  rotationAngle,
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
    if (error.includes('not found') || error.includes('404')) {
      return <UnknownStatus task={{ correlation_id: '', status: 'unknown', mandate: '', created_at: '', updated_at: '', max_ticks: 0 }} rotationAngle={rotationAngle} />;
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
  
  switch (status) {
    case 'in_queue':
    case 'pending':
      return <InQueueStatus correlationId={task.correlation_id} rotationAngle={rotationAngle} />;
    
    case 'in_progress':
    case 'accepted':
      return <InProgressStatus task={task} rotationAngle={rotationAngle} />;
    
    case 'completed':
      return <CompletedStatus task={task} />;
    
    case 'error':
    case 'failed':
      return <ErrorStatus task={task} />;
    
    default:
      if (task.tick !== undefined && task.max_ticks > 0) {
        return <InProgressStatus task={task} rotationAngle={rotationAngle} />;
      }
      return (
        <div className="p-4 rounded-xl bg-yellow-50 text-yellow-900 border-2 border-yellow-300">
          <div className="space-y-2 font-mono">
            <p>
              <span className="font-bold">Status:</span> {task.status || 'unknown'}
            </p>
            {task.tick !== undefined && (
              <p>
                <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
              </p>
            )}
            {task.error && (
              <p className="text-red-600">
                <span className="font-bold">Error:</span> {task.error}
              </p>
            )}
            <p className="text-xs mt-2 opacity-80">
              Unknown status state. Task may still be processing.
            </p>
          </div>
        </div>
      );
  }
}
