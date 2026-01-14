import React from 'react';
import { XCircle } from 'lucide-react';
import { TaskResponse } from '../../services/api';

interface ErrorStatusProps {
  task: TaskResponse;
}

export function ErrorStatus({ task }: ErrorStatusProps) {
  const progressPercent = task.tick !== undefined && task.max_ticks > 0
    ? Math.min(100, (task.tick / task.max_ticks) * 100)
    : 0;

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-xl bg-white text-black">
        {task.tick !== undefined && task.max_ticks > 0 && (
          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between text-sm font-mono">
              <span className="font-bold">Usage</span>
              <span>{task.tick} / {task.max_ticks} ticks</span>
            </div>
            <div className="h-6 bg-gray-200 rounded-full overflow-hidden relative">
              <div
                className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2"
                style={{
                  width: `${progressPercent}%`,
                  background: 'linear-gradient(to right, #ef4444, #dc2626)',
                }}
              >
                <span className="text-xs font-bold text-white">
                  {Math.round(progressPercent)}%
                </span>
              </div>
            </div>
          </div>
        )}
        <div className="h-1 rounded-full mb-4 bg-red-500" />
        <div className="space-y-2 font-mono">
          <div className="flex items-center gap-2">
            <XCircle size={20} className="text-red-600" />
            <p>
              <span className="font-bold">Status:</span> {task.status}
            </p>
          </div>
          {task.tick !== undefined && (
            <p>
              <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
            </p>
          )}
          {task.error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600">
                <span className="font-bold">Error:</span> {task.error}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
