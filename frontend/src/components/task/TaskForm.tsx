import React from 'react';

interface TaskFormProps {
  mandate: string;
  maxTicks: number;
  maxTicksInput: string;
  loading: boolean;
  onMandateChange: (mandate: string) => void;
  onMaxTicksChange: (value: string) => void;
  onMaxTicksBlur: () => void;
  onSubmit: () => void;
  onReset: () => void;
  hasTask: boolean;
}

export function TaskForm({
  mandate,
  maxTicks,
  maxTicksInput,
  loading,
  onMandateChange,
  onMaxTicksChange,
  onMaxTicksBlur,
  onSubmit,
  onReset,
  hasTask,
}: TaskFormProps) {
  const inputBg = 'bg-white text-black';

  return (
    <>
      <h2
        className="uppercase tracking-wider"
        style={{
          fontFamily: 'Impact, Arial Black, sans-serif',
          fontSize: '2rem',
          background: 'linear-gradient(to right, #ff00ff, #60a5fa, #38bdf8)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}
      >
        TASK SUBMISSION
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-white font-mono text-sm uppercase tracking-wide">
            Task Settings
          </h3>
          <div>
            <label className="block text-white font-mono mb-2">Mandate</label>
            <textarea
              placeholder="Enter your task mandate..."
              value={mandate}
              onChange={(e) => onMandateChange(e.target.value)}
              disabled={loading}
              className={`w-full p-4 rounded-xl ${inputBg} placeholder:text-gray-400 h-32 font-mono disabled:opacity-50`}
            />
          </div>
          <div>
            <label className="block text-white font-mono mb-2">
              Max ticks
            </label>
            <input
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              value={maxTicksInput}
              onChange={(e) => onMaxTicksChange(e.target.value)}
              onBlur={onMaxTicksBlur}
              disabled={loading}
              className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-400 font-mono disabled:opacity-50`}
            />
            <p className="font-mono text-xs mt-1" style={{ color: 'rgba(255, 255, 255, 0.8)' }}>
              Each user is restricted to 32 ticks per day across all tasks.
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        <button
          onClick={onSubmit}
          disabled={loading || !mandate.trim()}
          className="px-4 py-2 bg-white text-black rounded-xl border-2 border-gray-200 disabled:opacity-50
          font-mono active:scale-95 transition-all
          disabled:hover:bg-white disabled:hover:border-gray-200 disabled:active:scale-100"
          onMouseEnter={(e) => {
            if (!e.currentTarget.disabled) {
              e.currentTarget.style.backgroundColor = '#a855f7';
              e.currentTarget.style.color = 'white';
              e.currentTarget.style.borderColor = '#9333ea';
            }
          }}
          onMouseLeave={(e) => {
            if (!e.currentTarget.disabled) {
              e.currentTarget.style.backgroundColor = 'white';
              e.currentTarget.style.color = 'black';
              e.currentTarget.style.borderColor = '#e5e7eb';
            }
          }}
        >
          {loading ? 'Processing...' : 'Submit Task'}
        </button>
        {hasTask && (
          <button
            onClick={onReset}
            disabled={loading}
            className="px-6 py-4 bg-black text-white rounded-xl border-4 border-gray-800 font-mono
            hover:bg-red-600 hover:border-red-400 active:scale-95 transition-all disabled:opacity-50"
          >
            Reset
          </button>
        )}
      </div>
    </>
  );
}
