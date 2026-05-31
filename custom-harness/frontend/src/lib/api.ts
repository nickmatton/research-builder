// Thin fetch wrapper. Same-origin in prod, proxied to :7777 in dev.
import type {
  AgentsResponse,
  CascadePreview,
  ClaimsResponse,
  ComputeDetailResponse,
  ComputeListResponse,
  FilesResponse,
  Phase,
  PipelineStatus,
  RefinedSpecResponse,
  ReportResponse,
  SectionCritique,
  SectionDetail,
  SectionsResponse,
  SpecResponse,
  VerificationResponse,
  WorkspaceInfo,
} from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { credentials: "same-origin" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`GET ${path} → ${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function getText(path: string): Promise<string> {
  const res = await fetch(path, { credentials: "same-origin" });
  if (!res.ok) {
    throw new Error(`GET ${path} → ${res.status} ${res.statusText}`);
  }
  return res.text();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`POST ${path} → ${res.status} ${res.statusText}: ${t}`);
  }
  return res.json() as Promise<T>;
}

interface LaunchResponse {
  ok: boolean;
  workspace: string;
  name: string;
  paper_path: string;
  pid: number;
  log: string;
}

/** Error thrown by API helpers when a request returns non-2xx.
 *  ``body`` is the parsed JSON detail when available (FastAPI conventionally
 *  puts structured info under ``detail``), so callers can switch on
 *  ``body?.code`` instead of regex-matching message strings. */
export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  if (!res.ok) {
    let body: unknown = null;
    let message = `${res.status} ${res.statusText}`;
    try {
      const raw = await res.text();
      try {
        const parsed = JSON.parse(raw);
        body = parsed?.detail ?? parsed;
        if (typeof body === "string") message = body;
        else if (body && typeof body === "object" && "message" in body) {
          message = String((body as { message?: unknown }).message ?? message);
        }
      } catch {
        if (raw) message = raw;
      }
    } catch {
      // ignore
    }
    throw new ApiError(`POST ${path} → ${message}`, res.status, body);
  }
  return res.json() as Promise<T>;
}

export const api = {
  workspace: (): Promise<WorkspaceInfo> => get("/api/workspace"),
  launch: (
    file: File,
    name?: string,
    onConflict?: "fail" | "wipe" | "archive" | "resume",
    skipGates?: boolean,
    devMode?: boolean,
  ): Promise<LaunchResponse> => {
    const form = new FormData();
    form.append("paper", file, file.name);
    if (name) form.append("name", name);
    if (onConflict) form.append("on_conflict", onConflict);
    if (skipGates) form.append("skip_gates", "true");
    if (devMode !== undefined) form.append("dev_mode", devMode ? "true" : "false");
    return postForm<LaunchResponse>("/api/launch", form);
  },
  pipelineStatus: (): Promise<PipelineStatus> => get("/api/pipeline/status"),
  pipelineStop: (): Promise<{ ok: boolean; was_running: boolean }> =>
    post("/api/pipeline/stop", {}),
  spec: (): Promise<SpecResponse> => get("/api/spec"),
  phases: (): Promise<{ phases: Phase[] }> => get("/api/phases"),
  agents: (): Promise<AgentsResponse> => get("/api/agents"),
  files: (path = ""): Promise<FilesResponse> =>
    get(`/api/files?path=${encodeURIComponent(path)}`),
  fileText: (path: string): Promise<string> =>
    getText(`/api/file?path=${encodeURIComponent(path)}`),
  pdfUrl: (): string => "/api/pdf",
  commands: {
    chat: (text: string) =>
      post<{ ok: boolean; cmd_id: string }>("/api/commands/chat", { text }),
    forceRetry: (phase_id: string, rationale = "") =>
      post<{ ok: boolean; cmd_id: string }>("/api/commands/force_retry", {
        phase_id,
        rationale,
      }),
    injectNote: (text: string, phase_id?: string) =>
      post<{ ok: boolean; cmd_id: string }>("/api/commands/inject_note", {
        text,
        scope: phase_id ? "phase" : "global",
        phase_id,
      }),
  },
  refinedSpec: (phase_id: string): Promise<RefinedSpecResponse> =>
    get(`/api/refined-spec?phase_id=${encodeURIComponent(phase_id)}`),
  sections: (): Promise<SectionsResponse> => get("/api/sections"),
  section: (phase_id: string): Promise<SectionDetail> =>
    get(`/api/sections/${encodeURIComponent(phase_id)}`),
  sectionCritique: (phase_id: string): Promise<SectionCritique> =>
    get(`/api/sections/${encodeURIComponent(phase_id)}/critique`),
  claims: (): Promise<ClaimsResponse> => get("/api/claims"),
  verification: (phase_id: string): Promise<VerificationResponse> =>
    get(`/api/verification/${encodeURIComponent(phase_id)}`),
  report: (): Promise<ReportResponse> => get("/api/report"),
  compute: {
    list: (): Promise<ComputeListResponse> => get("/api/compute"),
    get: (instance_id: string): Promise<ComputeDetailResponse> =>
      get(`/api/compute/${encodeURIComponent(instance_id)}`),
  },
  spec_edits: {
    preview: (phase_id: string, content: string, before_agent = "builder") =>
      post<CascadePreview>("/api/spec/preview-edit", {
        phase_id,
        content,
        before_agent,
      }),
    apply: (
      phase_id: string,
      content: string,
      before_agent = "builder",
      rationale = "",
    ) =>
      post<{ ok: boolean; edit_cmd_id: string; jump_cmd_id: string }>(
        "/api/spec/apply-edit",
        { phase_id, content, before_agent, rationale },
      ),
  },
};
