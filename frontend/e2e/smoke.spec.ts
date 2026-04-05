import { expect, test } from "@playwright/test";

async function mockBackend(page: Parameters<typeof test>[0]["page"]) {
  await page.route("**/env/check", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        project_root: "/tmp/void-model",
        python: "/usr/bin/python",
        ffmpeg: "/usr/bin/ffmpeg",
        nvidia_smi: null,
        cuda: { available: false, error: null },
        gemini_api_key_set: false,
      }),
    });
  });

  await page.route("**/runs", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "[]",
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "run-smoke-1",
        workflow: "pass1_inference",
        status: "queued",
        created_at: new Date().toISOString(),
        started_at: null,
        ended_at: null,
        exit_code: null,
        command: [],
        name: null,
        params: {},
        log_path: "/tmp/fake.log",
        output_dir: null,
        error: null,
      }),
    });
  });

  await page.route("**/presets", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "[]",
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "preset-smoke-1",
        name: "Smoke",
        workflow: "pass1_inference",
        params: {},
        created_at: new Date().toISOString(),
      }),
    });
  });

  await page.route("**/validate/config", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        valid: true,
        errors: [],
        warnings: [],
        command_preview: ["python", "mock.py"],
      }),
    });
  });

  await page.route("**/data/sequences**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ root: "./sample", sequences: [] }),
    });
  });

  await page.route("**/artifacts**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ files: [] }),
    });
  });

  await page.route("**/cache/info**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ path: "./cache", exists: false, files: 0, bytes: 0 }),
    });
  });

  await page.route("**/cache/clear", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true, path: "./cache", exists: true, files: 0, bytes: 0 }),
    });
  });
}

test("loads dashboard header and environment panel", async ({ page }) => {
  await mockBackend(page);
  await page.goto("/");

  await expect(page.getByText("VOID Production Bay")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Environment Check" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Launch Workflow" })).toBeVisible();
});

test("can switch to pass2 workflow and see pass2 controls", async ({ page }) => {
  await mockBackend(page);
  await page.goto("/");

  await page.getByLabel("Workflow").selectOption("pass2_refine");
  await expect(page.getByPlaceholder("video names (comma-separated)")).toBeVisible();
  await expect(page.getByText("skip_noise_generation")).toBeVisible();
  await expect(page.getByText("use_quadmask")).toBeVisible();
});
