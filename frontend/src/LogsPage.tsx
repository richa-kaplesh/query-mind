/**
 * LogsPage.tsx
 *
 * Full-page route for the debug trace dashboard (/logs).
 * The Dashboard component is rendered as a standalone page rather than
 * an overlay. The "Close" button navigates back to the chat (/).
 */
import { useNavigate } from "react-router-dom";
import Dashboard from "./Dashboard";

export default function LogsPage() {
  const navigate = useNavigate();
  return <Dashboard onClose={() => navigate("/")} />;
}
