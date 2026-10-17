import { redirect } from "next/navigation";
import { createClient } from "../../../lib/supabase/server";
import AuthForm from "../AuthForm";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (user?.email_confirmed_at) redirect("/");
  const params = await searchParams;
  const errors: Record<string, string> = {
    callback:
      "That link has expired or could not be verified. Request a new link and open it in the browser where you started.",
    confirmation: "Confirm your email before opening your workspace.",
    oauth: "Google sign-in could not be completed. Please try again.",
    signout: "We couldn’t sign you out. Please try again.",
  };
  const error =
    typeof params.error === "string" ? errors[params.error] : undefined;
  const notice =
    params.notice === "password-updated"
      ? "Password updated. Sign in with your new password."
      : undefined;
  return <AuthForm initialError={error} initialMessage={notice} />;
}
