import React from 'react';
import { Github, ExternalLink } from 'lucide-react';

interface HeaderProps {
  text: string;
  workerCount: number | null;
}

export function Header({ text, workerCount }: HeaderProps) {
  return (
    <div className="mb-6 flex items-start justify-between">
      <div>
        <h1
          className="uppercase tracking-wider"
          style={{
            fontFamily: 'Impact, Arial Black, sans-serif',
            fontSize: '4.5rem',
            background: 'linear-gradient(to right, #ff00ff, #60a5fa, #38bdf8)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          EUGLENA
        </h1>
        <p className={`${text} mt-2 font-mono text-sm`}>
          Autonomous RAG agent for web automation and task execution
        </p>
        {workerCount !== null && (
          <p className={`${text} mt-1 font-mono text-xs opacity-70`}>
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
          <Github size={24} className={text} />
        </a>
        <a
          href="https://muksite.vercel.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 hover:opacity-70 transition-opacity"
        >
          <ExternalLink size={24} className={text} />
        </a>
      </div>
    </div>
  );
}
