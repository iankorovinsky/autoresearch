import { redirect } from "next/navigation";
import App from "../App";
import { createClient } from "../lib/supabase/server";

export default async function HomePage() {
  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  if (error || !user) redirect("/auth/login");
  if (!user.email_confirmed_at) redirect("/auth/login?error=confirmation");
  return (
    <App key={user.id} userId={user.id} email={user.email ?? "Researcher"} />
  );
}
