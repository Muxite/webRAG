import { createBrowserRouter } from "react-router";
import Auth from "@/app/pages/Auth";
import Dashboard from "@/app/pages/Dashboard";
import ResetPassword from "@/app/pages/ResetPassword";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Auth,
  },
  {
    path: "/login",
    Component: Auth,
  },
  {
    path: "/signup",
    Component: Auth,
  },
  {
    path: "/reset-password",
    Component: ResetPassword,
  },
  {
    path: "/dashboard",
    Component: Dashboard,
  },
]);