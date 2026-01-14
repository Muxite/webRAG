import React from 'react';

interface NoResponseStatusProps {
  error: string;
}

export function NoResponseStatus({ error }: NoResponseStatusProps) {
  return (
    <div className="p-6 rounded-xl bg-red-100 text-red-800 border-2 border-red-300">
      <p className="font-mono font-bold mb-2">No API Response</p>
      <p className="font-mono text-sm">{error}</p>
      <p className="font-mono text-xs mt-2 opacity-80">
        Please check your connection and ensure the API server is running.
      </p>
    </div>
  );
}
