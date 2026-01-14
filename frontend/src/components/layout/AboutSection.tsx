import React from 'react';

interface AboutSectionProps {
  dark: boolean;
  text: string;
}

export function AboutSection({ dark, text }: AboutSectionProps) {
  return (
    <div className={`mt-8 p-8 rounded-2xl ${
      dark ? 'bg-zinc-800 border-2 border-zinc-600' : 'bg-zinc-200 border-2 border-zinc-400'
    }`}>
      <h2
        className="uppercase tracking-wide text-lg mb-4"
        style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}
      >
        About Euglena
      </h2>
      <div className="space-y-4 font-mono text-sm">
        <p>
          Euglena is an autonomous RAG agent service that executes tasks through iterative reasoning, web interaction, and vector database storage. Tasks flow through a Gateway API, are consumed by Agent Workers via RabbitMQ, with status tracked in Redis and memory persisted in ChromaDB.
        </p>
        <p>
          The agent uses LLM-powered reasoning to break down tasks, perform web searches, visit URLs, and build up knowledge over time through persistent memory.
        </p>
        <div className="pt-4">
          <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">How the Scraper Works</h3>
          <p className="text-xs opacity-80 mb-2">
            When the agent needs to gather information from the web, it follows a two-step process:
          </p>
          <ul className="space-y-1 text-xs opacity-80">
            <li>
              • <span className="font-bold">Web Search:</span> Uses the Brave Search API to find relevant URLs based on the task mandate. The search returns titles, URLs, and descriptions that help the agent identify which pages to visit.
            </li>
            <li>
              • <span className="font-bold">Content Extraction:</span> When visiting a URL, the agent makes an HTTP GET request to fetch the HTML content. It then uses BeautifulSoup to parse the HTML, removing script and style tags, and extracts the main text content. This cleaned text is stored in the agent's observations and can be retrieved from ChromaDB for future reference.
            </li>
            <li>
              • <span className="font-bold">Memory Storage:</span> Extracted content is embedded and stored in ChromaDB, allowing the agent to recall relevant information across multiple ticks and tasks through semantic similarity search.
            </li>
          </ul>
        </div>
        <p className="opacity-80">
          <span className="font-bold">Usage Limits:</span> Each user is restricted to 32 ticks per day. This limit applies across all tasks submitted in a 24-hour period.
        </p>
        <div className="pt-4">
          <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Architecture</h3>
          <ul className="space-y-1 text-xs opacity-80">
            <li>
              • <span className="font-bold">Gateway:</span> FastAPI service accepting tasks via
              REST API
            </li>
            <li>
              • <span className="font-bold">Agent Workers:</span> Consume tasks from RabbitMQ
              and execute them
            </li>
            <li>
              • <span className="font-bold">Status Tracking:</span> Real-time monitoring in
              Redis
            </li>
            <li>
              • <span className="font-bold">Memory:</span> Context retention in ChromaDB vector
              database
            </li>
          </ul>
        </div>
        <div className="pt-4">
          <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Tech Stack</h3>
          <p className="text-xs opacity-80">
            Built with FastAPI, RabbitMQ, Redis, ChromaDB, and Docker. Powered by OpenAI and
            web search APIs.
          </p>
        </div>
        <div className="pt-4">
          <h3 className="uppercase text-xs tracking-wide mb-2 opacity-70">Attributions</h3>
          <ul className="space-y-1 text-xs opacity-80">
            <li>
              • UI components from <a href="https://ui.shadcn.com/" target="_blank" rel="noopener noreferrer" className="underline">shadcn/ui</a> used under <a href="https://github.com/shadcn-ui/ui/blob/main/LICENSE.md" target="_blank" rel="noopener noreferrer" className="underline">MIT license</a>
            </li>
            <li>
              • Icons from <a href="https://lucide.dev/" target="_blank" rel="noopener noreferrer" className="underline">Lucide</a> used under <a href="https://github.com/lucide-icons/lucide/blob/main/LICENSE" target="_blank" rel="noopener noreferrer" className="underline">ISC license</a>
            </li>
            <li>
              • Web scraping powered by <a href="https://www.crummy.com/software/BeautifulSoup/" target="_blank" rel="noopener noreferrer" className="underline">BeautifulSoup</a>
            </li>
            <li>
              • Search functionality via <a href="https://brave.com/search/api/" target="_blank" rel="noopener noreferrer" className="underline">Brave Search API</a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
