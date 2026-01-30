import React from 'react';
import { theme } from '../../config/theme';

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
}: TaskFormProps) {
  const inputBg = `${theme.colors.background.input} ${theme.colors.text.primary} ${theme.colors.border.input}`;

  return (
    <>
      <h2 className={`${theme.colors.text.primary} font-mono text-xl font-bold uppercase tracking-wide mb-4`}>
        TASK SUBMISSION
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className={`${theme.colors.text.primary} font-mono text-sm font-bold uppercase tracking-wide`}>
            Task Settings
          </h3>
          <div>
            <label className={`block ${theme.colors.text.primary} font-mono text-sm font-bold mb-2`}>Mandate</label>
            <textarea
              placeholder="Enter your task mandate..."
              value={mandate}
              onChange={(e) => onMandateChange(e.target.value)}
              disabled={loading}
              className={`w-full p-4 rounded-xl ${inputBg} placeholder:text-gray-500 h-32 font-mono text-sm disabled:opacity-50 border`}
            />
          </div>
          <div>
            <label className={`block ${theme.colors.text.primary} font-mono text-sm font-bold mb-2`}>
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
              className={`w-full p-3 rounded-xl ${inputBg} placeholder:text-gray-500 font-mono text-sm disabled:opacity-50 border`}
            />
            <p className={`font-mono text-xs mt-1 ${theme.colors.text.muted}`}>
              Each user is restricted to 32 ticks per day across all tasks.
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        <button
          onClick={onSubmit}
          disabled={loading || !mandate.trim()}
          className={`px-4 py-2 ${theme.colors.button.primary.bg} ${theme.colors.button.primary.text} rounded-xl border-2 ${theme.colors.button.primary.border} disabled:opacity-50
          font-mono text-sm font-bold active:scale-95 transition-all
          hover:bg-gray-200 disabled:hover:bg-gray-100 disabled:active:scale-100`}
        >
          {loading ? 'Processing...' : 'Submit Task'}
        </button>
        <button
          onClick={onReset}
          disabled={loading || !mandate.trim()}
          className={`px-4 py-2 ${theme.colors.button.secondary.bg} ${theme.colors.button.secondary.text} rounded-xl border-2 ${theme.colors.button.secondary.border} font-mono text-sm font-bold
          hover:bg-zinc-600 hover:border-zinc-500 active:scale-95 transition-all disabled:opacity-50`}
        >
          Clear
        </button>
      </div>
    </>
  );
}
