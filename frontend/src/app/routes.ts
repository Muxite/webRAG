import { createBrowserRouter } from "react-router";
import Home from "@/app/pages/Home";
import ResetPassword from "@/app/pages/ResetPassword";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Home,
  },
  {
    path: "/login",
    Component: Home,
  },
  {
    path: "/signup",
    Component: Home,
  },
  {
    path: "/reset-password",
    Component: ResetPassword,
  },
  {
    path: "/dashboard",
    Component: Home,
  },
]);