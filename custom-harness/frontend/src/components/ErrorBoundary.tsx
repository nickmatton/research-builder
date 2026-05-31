import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Optional label so error UI can say "Spec view crashed" etc. */
  label?: string;
}

interface State {
  error: Error | null;
}

/** Class-based ErrorBoundary so we can catch render-time errors in any
 *  subtree. React still doesn't ship a hooks-based equivalent in 2026. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string }) {
    // Surface in dev console; we don't ship telemetry from a local UI.
    console.error("[rb-ui]", this.props.label ?? "ErrorBoundary", error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
        <div className="text-sm font-medium text-[var(--color-fail)]">
          {this.props.label ? `${this.props.label} crashed` : "Something crashed"}
        </div>
        <pre className="max-w-md whitespace-pre-wrap break-words text-[10px] text-[var(--color-fg-dim)]">
          {this.state.error.message}
        </pre>
        <button
          type="button"
          onClick={this.reset}
          className="rounded bg-[var(--color-surface-2)] px-3 py-1 text-xs hover:bg-[var(--color-bg)]"
        >
          Reset
        </button>
      </div>
    );
  }
}
