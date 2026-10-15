import { redirect } from "next/navigation";
import { createClient } from "../../../lib/supabase/server";
import AuthForm from "../AuthForm";

export default async function ResetPasswordPage() {
  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  if (error || !user) redirect("/auth/login?error=callback");
  return <AuthForm resetPassword />;
}
