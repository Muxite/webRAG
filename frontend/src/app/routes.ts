import { createBrowserRouter } from "react-router";
import Login from "@/app/pages/Login";
import Dashboard from "@/app/pages/Dashboard";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Login,
  },
  {
    path: "/signup",
    // Redirect to login page (signup is now integrated)
    loader: () => {
      window.location.href = "/";
      return null;
    },
  },
  {
    path: "/dashboard",
    Component: Dashboard,
  },
]);