import React from 'react';
import { Github, ExternalLink } from 'lucide-react';
import { theme } from '../../config/theme';

interface HeaderProps {
  text: string;
  workerCount: number | null;
}

export function Header({ text, workerCount }: HeaderProps) {
  return (
    <div className="mb-6 flex items-start justify-between">
      <div>
        <h1 className={`${theme.colors.text.primary} font-mono text-4xl font-bold uppercase tracking-wider`}>
          EUGLENA
        </h1>
        <p className={`${theme.colors.text.primary} mt-2 font-mono text-sm`}>
          Autonomous RAG agent for web automation and task execution
        </p>
        {workerCount !== null && (
          <p className={`${theme.colors.text.primary} mt-1 font-mono text-xs ${theme.colors.text.muted}`}>
            Active workers: {workerCount}
          </p>
        )}
      </div>
      <div className="flex gap-3">
        <a
          href="https://github.com/Muxite/webRAG"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 hover:opacity-70 transition-opacity"
        >
          <Github size={24} className={theme.colors.text.primary} />
        </a>
        <a
          href="https://muksite.vercel.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 hover:opacity-70 transition-opacity"
        >
          <ExternalLink size={24} className={theme.colors.text.primary} />
        </a>
      </div>
    </div>
  );
}
