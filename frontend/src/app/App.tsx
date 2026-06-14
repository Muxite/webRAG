import { RouterProvider } from "react-router";
import { router } from "@/app/routes";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";

export default function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}
