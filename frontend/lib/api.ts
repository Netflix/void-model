import type { EnvCheck, RunRecord, Workflow } from "@/lib/types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }

  return (await res.json()) as T;
}

export async function getEnvCheck(): Promise<EnvCheck> {
  return request<EnvCheck>("/env/check");
}

export async function listRuns(): Promise<RunRecord[]> {
  return request<RunRecord[]>("/runs");
}

export async function createRun(payload: {
  workflow: Workflow;
  name?: string;
  params: Record<string, unknown>;
}): Promise<RunRecord> {
  return request<RunRecord>("/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function cancelRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/runs/${runId}/cancel`, {
    method: "POST",
  });
}

export async function getRunLogs(runId: string): Promise<{ lines: string[]; text: string }> {
  return request<{ lines: string[]; text: string }>(`/runs/${runId}/logs`);
}
