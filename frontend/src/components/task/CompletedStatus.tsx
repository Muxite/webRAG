import React from 'react';
import ReactMarkdown from 'react-markdown';
import { CheckCircle2 } from 'lucide-react';
import { TaskResponse } from '../../services/api';
import { theme } from '../../config/theme';

interface CompletedStatusProps {
  task: TaskResponse;
}

export function CompletedStatus({ task }: CompletedStatusProps) {
  const progressPercent = task.tick !== undefined && task.max_ticks > 0
    ? Math.min(100, (task.tick / task.max_ticks) * 100)
    : 0;

  return (
    <div className="space-y-4">
      <div className={`p-5 rounded-xl ${theme.colors.background.card} ${theme.colors.text.primary}`}>
        {task.tick !== undefined && task.max_ticks > 0 && (
          <div className="mb-5">
            <div className={`mb-3 flex items-center justify-between text-base font-mono`}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Usage</span>
              <span className={theme.colors.text.primary}>{task.tick} / {task.max_ticks} ticks</span>
            </div>
            <div className={`h-7 ${theme.colors.background.tertiary} rounded-full overflow-hidden relative`}>
              <div
                className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-3"
                style={{
                  width: `${progressPercent}%`,
                  background: 'linear-gradient(to right, #22c55e, #16a34a)',
                }}
              >
                <span className="text-sm font-bold text-white">
                  {Math.round(progressPercent)}%
                </span>
              </div>
            </div>
          </div>
        )}
        <div className="h-1.5 rounded-full mb-5 bg-green-500" />
        <div className="space-y-3 font-mono text-base">
          <div className="flex items-center gap-3">
            <CheckCircle2 size={24} className={theme.colors.status.completed.icon} />
            <p className={theme.colors.text.primary}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Status:</span> <span className={theme.colors.text.primary}>completed</span>
            </p>
          </div>
          {task.tick !== undefined && (
            <p className={theme.colors.text.primary}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Tick:</span> <span className={theme.colors.text.primary}>{task.tick} / {task.max_ticks}</span>
            </p>
          )}
          {task.result?.success !== undefined && (
            <p className={theme.colors.text.primary}>
              <span className={`font-bold ${theme.colors.text.primary}`}>Success:</span>{' '}
              <span className={task.result.success ? theme.colors.status.completed.icon : theme.colors.status.error.icon}>
                {task.result.success ? 'Yes' : 'No'}
              </span>
            </p>
          )}
        </div>
      </div>

      {task.result && (
        <div className="space-y-4">
          {task.result.deliverables && task.result.deliverables.length > 0 && (
            <div className={`p-6 rounded-xl ${theme.colors.status.completed.bg} ${theme.colors.status.completed.border} border-2`}>
              <h3 className={`font-mono font-bold mb-4 text-base ${theme.colors.status.completed.text} uppercase tracking-wide`}>
                Deliverables
              </h3>
              <div className={`markdown-content ${theme.colors.text.primary}`}>
                {task.result.deliverables.map((deliverable, index) => (
                  <div key={index}>
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => <p className={`mb-3 font-mono text-base ${theme.colors.text.primary} leading-relaxed`}>{children}</p>,
                        h1: ({ children }) => <h1 className={`text-xl font-bold mb-3 mt-4 font-mono ${theme.colors.text.primary}`}>{children}</h1>,
                        h2: ({ children }) => <h2 className={`text-lg font-bold mb-3 mt-4 font-mono ${theme.colors.text.primary}`}>{children}</h2>,
                        h3: ({ children }) => <h3 className={`text-base font-bold mb-2 mt-3 font-mono ${theme.colors.text.primary}`}>{children}</h3>,
                        ul: ({ children }) => <ul className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ul>,
                        ol: ({ children }) => <ol className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ol>,
                        li: ({ children }) => <li className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ marginBottom: '0.5rem' }}>{children}</li>,
                        code: ({ children }) => <code className={`${theme.colors.markdown.code.bg} px-1.5 py-0.5 rounded font-mono text-sm ${theme.colors.markdown.code.text}`}>{children}</code>,
                        pre: ({ children }) => <pre className={`${theme.colors.markdown.pre.bg} p-3 rounded overflow-x-auto mb-3 font-mono text-sm ${theme.colors.markdown.pre.text} leading-relaxed`}>{children}</pre>,
                        strong: ({ children }) => <strong className={`font-bold font-mono ${theme.colors.text.primary}`}>{children}</strong>,
                        em: ({ children }) => <em className={`italic font-mono ${theme.colors.text.primary}`}>{children}</em>,
                        a: ({ href, children }) => <a href={href} className="text-blue-400 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
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
            <div className={`p-6 rounded-xl ${theme.colors.status.inProgress.bg} ${theme.colors.status.inProgress.border} border-2`}>
              <h3 className={`font-mono font-bold mb-4 text-base ${theme.colors.status.inProgress.text} uppercase tracking-wide`}>
                Notes
              </h3>
              <div className={`markdown-content ${theme.colors.text.primary}`}>
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className={`mb-3 font-mono text-base ${theme.colors.text.primary} leading-relaxed`}>{children}</p>,
                    h1: ({ children }) => <h1 className={`text-xl font-bold mb-3 mt-4 font-mono ${theme.colors.text.primary}`}>{children}</h1>,
                    h2: ({ children }) => <h2 className={`text-lg font-bold mb-3 mt-4 font-mono ${theme.colors.text.primary}`}>{children}</h2>,
                    h3: ({ children }) => <h3 className={`text-base font-bold mb-2 mt-3 font-mono ${theme.colors.text.primary}`}>{children}</h3>,
                    ul: ({ children }) => <ul className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ listStyle: 'disc', paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ul>,
                    ol: ({ children }) => <ol className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ listStyle: 'decimal', paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ol>,
                    li: ({ children }) => <li className={`font-mono text-base ${theme.colors.text.primary} leading-relaxed`} style={{ marginBottom: '0.5rem' }}>{children}</li>,
                    code: ({ children }) => <code className={`${theme.colors.markdown.code.bg} px-1.5 py-0.5 rounded font-mono text-sm ${theme.colors.markdown.code.text}`}>{children}</code>,
                    pre: ({ children }) => <pre className={`${theme.colors.markdown.pre.bg} p-3 rounded overflow-x-auto mb-3 font-mono text-sm ${theme.colors.markdown.pre.text} leading-relaxed`}>{children}</pre>,
                    strong: ({ children }) => <strong className={`font-bold font-mono ${theme.colors.text.primary}`}>{children}</strong>,
                    em: ({ children }) => <em className={`italic font-mono ${theme.colors.text.primary}`}>{children}</em>,
                    a: ({ href, children }) => <a href={href} className="text-blue-400 underline font-mono" target="_blank" rel="noopener noreferrer">{children}</a>,
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
