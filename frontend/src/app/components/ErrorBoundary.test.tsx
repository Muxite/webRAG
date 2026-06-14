import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function Boom({ shouldThrow }: { shouldThrow: boolean }): JSX.Element {
  if (shouldThrow) {
    throw new Error("boom from child");
  }
  return <div>child rendered</div>;
}

describe("ErrorBoundary", () => {
  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>
    );
    expect(screen.getByText("child rendered")).toBeInTheDocument();
  });

  it("renders default fallback on render error", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    expect(screen.getByText("boom from child")).toBeInTheDocument();
    errorSpy.mockRestore();
  });

  it("invokes custom fallback when supplied", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const fallback = vi.fn((err: Error) => <div>custom: {err.message}</div>);
    render(
      <ErrorBoundary fallback={fallback}>
        <Boom shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText(/custom: boom from child/)).toBeInTheDocument();
    expect(fallback).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("resets via reset button so children can re-render", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    function Toggleable() {
      return <Boom shouldThrow={false} />;
    }
    const { rerender } = render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Try again/i }));

    rerender(
      <ErrorBoundary>
        <Toggleable />
      </ErrorBoundary>
    );
    expect(screen.getByText("child rendered")).toBeInTheDocument();
    errorSpy.mockRestore();
  });
});
