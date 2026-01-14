import React from 'react';
import { AlertCircle } from 'lucide-react';

interface OutOfTicksStatusProps {
  remaining: number;
  error: string;
}

export function OutOfTicksStatus({ remaining, error }: OutOfTicksStatusProps) {
  return (
    <div className="p-6 rounded-xl bg-orange-100 text-orange-900 border-2 border-orange-400">
      <div className="flex items-start gap-3">
        <AlertCircle size={24} className="flex-shrink-0 mt-1" />
        <div className="flex-1">
          <p className="font-mono font-bold mb-2">Daily Tick Limit Exceeded</p>
          <p className="font-mono text-sm mb-2">{error}</p>
          <div className="mt-4 p-3 bg-orange-50 rounded-lg border border-orange-200">
            <p className="font-mono text-sm">
              <span className="font-bold">Remaining ticks today:</span> {remaining}
            </p>
            <p className="font-mono text-xs mt-2 opacity-80">
              Your daily limit of 32 ticks has been reached. Please try again tomorrow or reduce the max_ticks for your task.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
