import { useState } from "react";
import VectorBox from "./VectorBox";
import { Clock, Activity, ChevronDown, ChevronUp, X } from "lucide-react";
import * as Progress from "@radix-ui/react-progress";

interface Task {
  id: string;
  timestamp: string;
  completedAt?: string | null;
  mandate: string;
  maxTicks: number;
  status: string;
  ticksUsed: number;
  results: string;
  deliverables?: string;
  notes?: string;
}

interface TaskContainerProps {
  task: Task;
  onDelete?: (taskId: string) => void;
  defaultOpen?: boolean;
  themeColors?: {
    primary: string;
    secondary: string;
    text: string;
    textMuted: string;
    surface: string;
    boxBg: string;
  };
}

export default function TaskContainer({ task, onDelete, defaultOpen, themeColors }: TaskContainerProps) {
  const [isExpanded, setIsExpanded] = useState(Boolean(defaultOpen));

  const statusColor = {
    completed: {
      border: "#10B981",
      text: "text-green-400",
    },
    in_progress: {
      border: "#F59E0B",
      text: "text-yellow-400",
    },
    pending: {
      border: "#F59E0B",
      text: "text-yellow-400",
    },
    accepted: {
      border: "#F59E0B",
      text: "text-yellow-400",
    },
    running: {
      border: "#F59E0B",
      text: "text-yellow-400",
    },
    failed: {
      border: "#EF4444",
      text: "text-red-400",
    },
    error: {
      border: "#EF4444",
      text: "text-red-400",
    },
  }[task.status.toLowerCase()] || {
    border: "#6B7280",
    text: "text-gray-400",
  };

  const progressPercentage = task.maxTicks > 0
    ? Math.min(100, Math.max(0, (task.ticksUsed / task.maxTicks) * 100))
    : 0;

  const primaryColor = themeColors?.primary || "#22D3EE";
  const secondaryColor = themeColors?.secondary || "#FF3D9A";
  const boxBg = themeColors?.boxBg || "#0a0a0aD9";

  return (
    <VectorBox padding={6} borderColor={themeColors?.secondary || statusColor.border} bgColor={boxBg}>
      <div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-left"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 sm:gap-3">
              <Clock className="w-3 h-3 sm:w-4 sm:h-4" style={{ color: primaryColor }} />
              <span className="text-metadata-muted">{task.timestamp}</span>
            </div>
            <div className="flex items-center gap-2 sm:gap-3">
              <span className={`text-status px-2 py-1 border ${statusColor.text}`} style={{ borderColor: statusColor.border }}>
                {task.status.toUpperCase()}
              </span>
              {onDelete && (
                <button
                  type="button"
                  aria-label="Delete task"
                  onClick={(event) => {
                    event.stopPropagation();
                    onDelete(task.id);
                  }}
                  className="border px-1.5 py-1 transition-colors"
                  style={{ borderColor: `${primaryColor}40`, color: primaryColor }}
                >
                  <X className="w-3 h-3 sm:w-4 sm:h-4" />
                </button>
              )}
              {isExpanded ? (
                <ChevronUp className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: primaryColor }} />
              ) : (
                <ChevronDown className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: primaryColor }} />
              )}
            </div>
          </div>

          <div className="mb-3">
            <span className="text-body">{task.mandate}</span>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-metadata-secondary">Iterations Used</span>
              <span className="text-metadata-primary">
                {task.ticksUsed} / {task.maxTicks}
              </span>
            </div>
            <Progress.Root
              className="relative overflow-hidden border"
              style={{ 
                height: 10,
                backgroundColor: `${primaryColor}20`,
                borderColor: primaryColor
              }}
              value={progressPercentage}
            >
              <Progress.Indicator
                className="h-full transition-all duration-500 ease-out"
                style={{
                  transform: `translateX(-${100 - progressPercentage}%)`,
                  background: `linear-gradient(to right, ${primaryColor}, ${secondaryColor})`,
                }}
              />
            </Progress.Root>
          </div>
        </button>

        {isExpanded && (task.results || task.deliverables || task.notes || task.completedAt) && (
          <div
            className="mt-4 pt-4 border-t"
            style={{ borderColor: `${primaryColor}30` }}
          >
            <div className="border p-3 sm:p-4" style={{ borderColor: `${primaryColor}30` }}>
              {task.completedAt && (
                <div className="text-metadata-muted mb-3">
                  Completed: {new Date(task.completedAt).toLocaleString()}
                </div>
              )}
              {task.deliverables && (
                <div className="mb-4">
                  <div className="text-metadata-secondary mb-2">Deliverables</div>
                  <pre className="text-body whitespace-pre-wrap">{task.deliverables}</pre>
                </div>
              )}
              {task.notes && (
                <div className="mb-4">
                  <div className="text-metadata-secondary mb-2">Notes</div>
                  <pre className="text-body whitespace-pre-wrap">{task.notes}</pre>
                </div>
              )}
              {!task.deliverables && !task.notes && task.results && (
                <pre className="text-body whitespace-pre-wrap">
                  {task.results}
                </pre>
              )}
            </div>
          </div>
        )}
      </div>
    </VectorBox>
  );
}