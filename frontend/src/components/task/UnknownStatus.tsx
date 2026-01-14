import React from 'react';
import { HelpCircle } from 'lucide-react';
import { TaskResponse } from '../../services/api';

interface UnknownStatusProps {
  task: TaskResponse;
  rotationAngle: number;
}

export function UnknownStatus({ task, rotationAngle }: UnknownStatusProps) {
  return (
    <div className="p-4 rounded-xl bg-yellow-50 border-2 border-yellow-200">
      <div className="flex items-start gap-3">
        <HelpCircle size={20} className="text-yellow-600 flex-shrink-0 mt-1" />
        <div className="flex-1">
          <p className="font-mono font-bold text-yellow-900 mb-2">
            Waiting for Response from Backend
          </p>
          <p className="font-mono text-xs text-yellow-700 mb-3">
            The task status is not yet available. This may happen if the task was just submitted or if there's a delay in processing.
          </p>
          {task.correlation_id && (
            <p className="font-mono text-xs text-yellow-600">
              Correlation ID: {task.correlation_id.slice(0, 8)}...
            </p>
          )}
          <div className="mt-3 flex items-center gap-2">
            <div className="relative w-6 h-6 flex items-center justify-center">
              <div
                className="absolute w-1 h-4 bg-yellow-600 rounded-full origin-bottom"
                style={{
                  transform: `rotate(${rotationAngle}deg)`,
                }}
              />
            </div>
            <span className="font-mono text-xs text-yellow-600">Checking status...</span>
          </div>
        </div>
      </div>
    </div>
  );
}
