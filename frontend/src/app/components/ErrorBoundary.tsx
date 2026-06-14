import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }
      return (
        <div
          role="alert"
          style={{
            padding: "32px",
            margin: "32px auto",
            maxWidth: "640px",
            border: "2px solid #ef4444",
            background: "#1a1a1a",
            color: "#f5f5f5",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <h1 style={{ marginTop: 0, color: "#ef4444" }}>Something went wrong</h1>
          <p style={{ opacity: 0.8 }}>
            The interface hit an unexpected error. Try reloading the page.
          </p>
          <pre
            style={{
              padding: "12px",
              background: "#0a0a0a",
              border: "1px solid #333",
              overflow: "auto",
              fontSize: "12px",
              whiteSpace: "pre-wrap",
            }}
          >
            {this.state.error.message}
          </pre>
          <button
            type="button"
            onClick={this.reset}
            style={{
              marginTop: "12px",
              padding: "8px 16px",
              background: "#ef4444",
              color: "#fff",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
