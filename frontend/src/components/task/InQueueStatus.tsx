import React from 'react';
import { Clock } from 'lucide-react';

interface InQueueStatusProps {
  correlationId: string;
  rotationAngle: number;
}

export function InQueueStatus({ correlationId, rotationAngle }: InQueueStatusProps) {
  return (
    <div className="p-4 rounded-xl bg-blue-50 border-2 border-blue-200">
      <div className="flex items-center gap-3">
        <div className="relative w-8 h-8 flex items-center justify-center">
          <Clock size={20} className="text-blue-600" />
          <div
            className="absolute w-1 h-6 bg-blue-600 rounded-full origin-bottom"
            style={{
              transform: `rotate(${rotationAngle}deg)`,
            }}
          />
        </div>
        <div className="flex-1">
          <p className="font-mono font-bold text-blue-900">Task in Queue</p>
          <p className="font-mono text-xs text-blue-700 mt-1">
            Your task has been submitted and is waiting to be picked up by a worker.
          </p>
          <p className="font-mono text-xs text-blue-600 mt-1 opacity-75">
            ID: {correlationId.slice(0, 8)}...
          </p>
        </div>
      </div>
    </div>
  );
}
