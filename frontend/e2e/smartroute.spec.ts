import { expect, test, type Page } from "@playwright/test";

const activeDocument = {
  id: "doc-1",
  filename: "Cover Letter.pdf",
  content_type: "application/pdf",
  size_bytes: 1024,
  storage_bucket: "documents",
  storage_path: "tenant-1/Cover Letter.pdf",
  created_at: "2026-09-28T10:00:00Z",
};

async function mockBootstrap(page: Page) {
  await page.route("**/v1/auth/demo-token", async (route) => {
    await route.fulfill({
      json: {
        access_token: "e2e-token",
        token_type: "bearer",
        expires_at: 4102444800,
        session_id: "e2e-session",
      },
    });
  });
  await page.route("**/v1/documents", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({ json: { documents: [activeDocument], total: 1 } });
      return;
    }
    await route.continue();
  });
}

test.beforeEach(async ({ page }) => {
  await mockBootstrap(page);
});

test("Quality First RAG answer shows only active document citations", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.startsWith("mobile"), "desktop sidebar workflow");
  let requestPayload: Record<string, unknown> = {};
  await page.route("**/v1/query/stream", async (route) => {
    requestPayload = route.request().postDataJSON();
    const citation = {
      id: "C1",
      filename: "Cover Letter.pdf",
      page: 1,
      section: null,
      excerpt: "Applying for the Backend Engineer role with Python API experience.",
    };
    const answer = "The applicant is applying for a Backend Engineer role. [C1]";
    const events = [
      { type: "metadata", data: { sources: [], citations: [citation] } },
      { type: "chunk", content: answer },
      {
        type: "done",
        result: {
          answer,
          model_used: "openai/gpt-oss-120b",
          sources: [],
          citations: [citation],
          success: true,
        },
      },
    ];
    await route.fulfill({
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.goto("/");
  await expect(page.getByText("1 doc embedded")).toBeVisible();
  await expect(page.getByText("Cover Letter.pdf")).toBeVisible();
  await expect(page.getByText("SQL Mentor.txt")).toHaveCount(0);

  await page.getByRole("button", { name: "RAG" }).click();
  await page.getByRole("button", { name: "Quality First" }).click();
  await page.getByPlaceholder("Type your message here...").fill("What is this cover letter about?");
  await page.getByPlaceholder("Type your message here...").press("Enter");

  await expect(page.getByText(/applying for a Backend Engineer role/)).toBeVisible();
  const citationButton = page.getByRole("button", {
    name: "Citation 1: Cover Letter.pdf, page 1",
    exact: true,
  });
  await expect(citationButton).toBeVisible();
  await citationButton.click();
  await expect(page.getByText("Page 1")).toBeVisible();
  await expect(page.getByText(/Python API experience/)).toBeVisible();

  expect(requestPayload.strategy).toBe("quality_first");
  expect(requestPayload.use_retrieval).toBe(true);
});

test("terminal streaming failure replaces partial output and removes citations", async ({ page }) => {
  await page.route("**/v1/query/stream", async (route) => {
    const events = [
      { type: "chunk", content: "Partial provider output [C1]" },
      { type: "replace", content: "Request failed. Please try again." },
      {
        type: "done",
        result: {
          answer: "Request failed. Please try again.",
          sources: [],
          citations: [],
          success: false,
          error: "pipeline_error",
        },
      },
    ];
    await route.fulfill({
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("Type your message here...").fill("Trigger failure");
  await page.getByPlaceholder("Type your message here...").press("Enter");

  await expect(page.getByText("Request failed. Please try again.")).toBeVisible();
  await expect(page.getByText("Partial provider output")).toHaveCount(0);
  await expect(page.getByLabel("Answer citations")).toHaveCount(0);
});

test("mobile interface remains within the viewport", async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.startsWith("mobile"), "mobile-only assertion");
  await page.goto("/");

  await expect(page.getByPlaceholder("Type your message here...")).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
