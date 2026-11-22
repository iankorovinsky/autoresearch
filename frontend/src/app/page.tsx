import { redirect } from "next/navigation";
import App, { type LocalAgent } from "../App";
import { createClient } from "../lib/supabase/server";

export default async function HomePage() {
  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  if (error || !user) redirect("/auth/login");
  if (!user.email_confirmed_at) redirect("/auth/login?error=confirmation");
  const agents: LocalAgent[] = [];
  if (process.env.AUTORESEARCH_CLAUDE_ENABLED === "1") agents.push("claude");
  if (process.env.AUTORESEARCH_CODEX_ENABLED === "1") agents.push("codex");

  return (
    <App
      key={user.id}
      userId={user.id}
      email={user.email ?? "Researcher"}
      agents={agents}
    />
  );
}
