import { useEffect, useState } from "react";
import { Markdown } from "./Markdown";
import { api } from "../lib/api";
import { getEventStream } from "../lib/events";
import type { ReportResponse } from "../lib/types";

export function ReportView() {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = () => {
    api.report().then(setReport).catch((e: Error) => setErr(e.message));
  };

  useEffect(() => {
    reload();
    const off = getEventStream().subscribe((e) => {
      // Report appears at pipeline end; the artifact_created event with
      // artifact_type=reproduction_report is the cleanest trigger, but
      // run_completed is a reliable fallback.
      if (
        e.type === "run_completed" ||
        (e.type === "artifact_created" &&
          (e as { artifact_type?: string }).artifact_type === "reproduction_report")
      ) {
        reload();
      }
    });
    return off;
  }, []);

  if (err) {
    return <div className="p-4 text-sm text-[var(--color-fail)]">{err}</div>;
  }
  if (!report) {
    return <div className="p-4 text-sm text-[var(--color-fg-dim)]">Loading…</div>;
  }
  if (!report.exists) {
    return (
      <div className="p-4 text-sm text-[var(--color-fg-dim)]">
        No reproduction report yet. The report is written at pipeline end.
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto">
      <div className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-fg-dim)]">
          Final reproduction report
        </div>
        <div className="mt-0.5 text-xs font-mono text-[var(--color-fg-muted)]">
          {report.path}
        </div>
      </div>
      <div className="prose-spec px-4 py-3 text-xs text-[var(--color-fg)]">
        <Markdown>{report.content ?? ""}</Markdown>
      </div>
    </div>
  );
}
