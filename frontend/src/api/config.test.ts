import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("/utils/supabase/info.tsx", () => ({
  projectId: "test-project",
  publicAnonKey: "anon-key",
}));

vi.mock("@/lib/supabase", () => ({
  supabase: {
    auth: { getUser: vi.fn(), getSession: vi.fn() },
    from: vi.fn(),
  },
}));

describe("fetchWorkerCount mock-fallback gating", () => {
  let originalFetch: typeof global.fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
    vi.resetModules();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("returns server worker count on success", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ activeWorkers: 7 }),
    }) as unknown as typeof global.fetch;
    const { fetchWorkerCount } = await import("./config");
    expect(await fetchWorkerCount()).toBe(7);
  });

  it("falls back to mock when fetch throws and fallback is enabled (default)", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("network")) as unknown as typeof global.fetch;
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { fetchWorkerCount } = await import("./config");
    const count = await fetchWorkerCount();
    expect(typeof count).toBe("number");
    expect(count).toBeGreaterThanOrEqual(0);
    errSpy.mockRestore();
  });

  it("re-throws fetch errors when VITE_DISABLE_MOCK_FALLBACK=true", async () => {
    vi.stubEnv("VITE_DISABLE_MOCK_FALLBACK", "true");
    global.fetch = vi.fn().mockRejectedValue(new Error("network down")) as unknown as typeof global.fetch;
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { fetchWorkerCount } = await import("./config");
    await expect(fetchWorkerCount()).rejects.toThrow("network down");
    errSpy.mockRestore();
  });
});
