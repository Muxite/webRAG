import React from 'react';
import { Loader2 } from 'lucide-react';
import { TaskResponse } from '../../services/api';

interface InProgressStatusProps {
  task: TaskResponse;
  rotationAngle: number;
}

export function InProgressStatus({ task, rotationAngle }: InProgressStatusProps) {
  const progressPercent = task.tick !== undefined && task.max_ticks > 0
    ? Math.min(100, (task.tick / task.max_ticks) * 100)
    : 0;

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-xl bg-white flex items-center gap-3">
        <div className="relative w-8 h-8 flex items-center justify-center">
          <Loader2 size={20} className="text-blue-600 animate-spin" />
          <div
            className="absolute w-1 h-6 bg-blue-600 rounded-full origin-bottom"
            style={{
              transform: `rotate(${rotationAngle}deg)`,
            }}
          />
        </div>
        <div className="text-xs font-mono text-gray-600">
          Processing task...
        </div>
      </div>
      
      <div className="p-4 rounded-xl bg-white text-black">
        {task.tick !== undefined && task.max_ticks > 0 && (
          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between text-sm font-mono">
              <span className="font-bold">Progress</span>
              <span>{task.tick} / {task.max_ticks} ticks</span>
            </div>
            <div className="h-6 bg-gray-200 rounded-full overflow-hidden relative">
              <div
                className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2 animate-pulse"
                style={{
                  width: `${progressPercent}%`,
                  background: 'linear-gradient(to right, #3b82f6, #a855f7)',
                }}
              >
                <span className="text-xs font-bold text-white">
                  {Math.round(progressPercent)}%
                </span>
              </div>
            </div>
          </div>
        )}
        <div className="h-1 rounded-full mb-4 bg-blue-500" />
        <div className="space-y-2 font-mono">
          <p>
            <span className="font-bold">Status:</span> {task.status || 'in_progress'}
          </p>
          {task.tick !== undefined && task.max_ticks > 0 && (
            <p>
              <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
            </p>
          )}
          {task.correlation_id && (
            <p className="text-xs opacity-75">
              <span className="font-bold">ID:</span> {task.correlation_id.slice(0, 8)}...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
