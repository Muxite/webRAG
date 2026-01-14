import React from 'react';
import ReactMarkdown from 'react-markdown';
import { CheckCircle2 } from 'lucide-react';
import { TaskResponse } from '../../services/api';

interface CompletedStatusProps {
  task: TaskResponse;
}

export function CompletedStatus({ task }: CompletedStatusProps) {
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
                  background: 'linear-gradient(to right, #22c55e, #16a34a)',
                }}
              >
                <span className="text-xs font-bold text-white">
                  {Math.round(progressPercent)}%
                </span>
              </div>
            </div>
          </div>
        )}
        <div className="h-1 rounded-full mb-4 bg-green-500" />
        <div className="space-y-2 font-mono">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={20} className="text-green-600" />
            <p>
              <span className="font-bold">Status:</span> completed
            </p>
          </div>
          {task.tick !== undefined && (
            <p>
              <span className="font-bold">Tick:</span> {task.tick} / {task.max_ticks}
            </p>
          )}
          {task.result?.success !== undefined && (
            <p>
              <span className="font-bold">Success:</span>{' '}
              <span className={task.result.success ? 'text-green-600' : 'text-red-600'}>
                {task.result.success ? 'Yes' : 'No'}
              </span>
            </p>
          )}
        </div>
      </div>

      {task.result && (
        <div className="space-y-4">
          {task.result.deliverables && task.result.deliverables.length > 0 && (
            <div className="p-6 rounded-xl bg-green-50 border-2 border-green-200">
              <h3 className="font-mono font-bold mb-4 text-green-900 uppercase tracking-wide">
                Deliverables
              </h3>
              <div className="space-y-4 text-gray-800">
                {task.result.deliverables.map((deliverable, index) => (
                  <div key={index} className="markdown-content">
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => <p className="mb-2 font-mono">{children}</p>,
                        h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-4 font-mono">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 font-mono">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-2 font-mono">{children}</h3>,
                        ul: ({ children }) => <ul className="font-mono" style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ul>,
                        ol: ({ children }) => <ol className="font-mono" style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ol>,
                        li: ({ children }) => <li className="font-mono" style={{ marginBottom: '0.25rem' }}>{children}</li>,
                        code: ({ children }) => <code className="bg-gray-200 px-1 rounded font-mono text-xs">{children}</code>,
                        pre: ({ children }) => <pre className="bg-gray-200 p-2 rounded overflow-x-auto mb-2 font-mono text-xs">{children}</pre>,
                        strong: ({ children }) => <strong className="font-bold font-mono">{children}</strong>,
                        em: ({ children }) => <em className="italic font-mono">{children}</em>,
                        a: ({ href, children }) => <a href={href} className="text-blue-600 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
                      }}
                    >
                      {deliverable}
                    </ReactMarkdown>
                  </div>
                ))}
              </div>
            </div>
          )}

          {task.result.notes && (
            <div className="p-6 rounded-xl bg-blue-50 border-2 border-blue-200">
              <h3 className="font-mono font-bold mb-4 text-blue-900 uppercase tracking-wide">
                Notes
              </h3>
              <div className="text-gray-800 markdown-content">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 font-mono">{children}</p>,
                    h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-4 font-mono">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 font-mono">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-2 font-mono">{children}</h3>,
                    ul: ({ children }) => <ul className="font-mono" style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ul>,
                    ol: ({ children }) => <ol className="font-mono" style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.5rem' }}>{children}</ol>,
                    li: ({ children }) => <li className="font-mono" style={{ marginBottom: '0.25rem' }}>{children}</li>,
                    code: ({ children }) => <code className="bg-gray-200 px-1 rounded font-mono text-xs">{children}</code>,
                    pre: ({ children }) => <pre className="bg-gray-200 p-2 rounded overflow-x-auto mb-2 font-mono text-xs">{children}</pre>,
                    strong: ({ children }) => <strong className="font-bold font-mono">{children}</strong>,
                    em: ({ children }) => <em className="italic font-mono">{children}</em>,
                    a: ({ href, children }) => <a href={href} className="text-blue-600 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
                  }}
                >
                  {task.result.notes}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
