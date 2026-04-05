import { afterEach, describe, expect, it, vi } from "vitest";

import { createRun, getEnvCheck, listArtifacts, updatePromptBg, validateConfig } from "@/lib/api";

describe("lib/api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("calls backend with default URL and no-store cache", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await getEnvCheck();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/env/check",
      expect.objectContaining({
        cache: "no-store",
      }),
    );
  });

  it("throws string detail errors from backend", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "bad request" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      validateConfig({
        workflow: "pass1_inference",
        params: {},
      }),
    ).rejects.toThrow("bad request");
  });

  it("throws object detail errors from backend", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: { errors: ["missing file"] } }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      createRun({
        workflow: "pass1_inference",
        params: {},
      }),
    ).rejects.toThrow('{"errors":["missing file"]}');
  });

  it("returns artifact list payload wrapper correctly", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          files: [{ relative: "out/video.mp4", path: "/tmp/video.mp4", size_bytes: 1234 }],
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const files = await listArtifacts("run 1");
    expect(files).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/artifacts?runId=run%201",
      expect.any(Object),
    );
  });

  it("sends prompt update body with snake_case keys", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await updatePromptBg("./sample/lime", "new bg");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/data/prompt",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          sequence_path: "./sample/lime",
          bg: "new bg",
        }),
      }),
    );
  });
});
