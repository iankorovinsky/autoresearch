"use server";

import { redirect } from "next/navigation";
import { createClient } from "../../lib/supabase/server";
import { appOrigin } from "../../lib/auth/config";
import {
  credentials,
  validEmail,
  validPassword,
} from "../../lib/auth/validation";

export type AuthState = {
  error?: string;
  message?: string;
  verification?: boolean;
};

function deliveryError(code?: string): AuthState {
  if (
    code === "over_email_send_rate_limit" ||
    code === "over_request_rate_limit"
  ) {
    return { error: "Too many requests. Please wait before trying again." };
  }
  if (code === "email_address_not_authorized") {
    return {
      error:
        "Email delivery is limited to project-team addresses until an email provider is connected.",
    };
  }
  return { error: "We couldn’t send that email. Please try again later." };
}

export async function signIn(
  _previous: AuthState,
  form: FormData,
): Promise<AuthState> {
  const { email, password } = credentials(form);
  if (!validEmail(email) || !password)
    return { error: "Enter your email address and password." };
  const supabase = await createClient();
  const { error } = await supabase.auth.signInWithPassword({ email, password });
  if (error?.code === "email_not_confirmed")
    return {
      error: "Confirm your email before signing in.",
      verification: true,
    };
  if (error)
    return { error: "Unable to sign in. Check your details and try again." };
  redirect("/");
}

export async function signUp(
  _previous: AuthState,
  form: FormData,
): Promise<AuthState> {
  const { email, password } = credentials(form);
  if (!validEmail(email)) return { error: "Enter a valid email address." };
  if (!validPassword(password))
    return { error: "Use a password between 12 and 128 characters." };
  const supabase = await createClient();
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: { emailRedirectTo: `${appOrigin()}/auth/callback` },
  });
  if (error) {
    if (error.code === "weak_password")
      return { error: "Choose a stronger password." };
    return deliveryError(error.code);
  }
  // Do not admit a password signup if confirmation was accidentally disabled.
  if (data.session) await supabase.auth.signOut({ scope: "local" });
  return {
    message:
      "Check your inbox for a confirmation link. If this address is already registered, sign in or reset your password.",
    verification: true,
  };
}

export async function resendConfirmation(
  _previous: AuthState,
  form: FormData,
): Promise<AuthState> {
  const { email } = credentials(form);
  if (!validEmail(email))
    return { error: "Enter your email address to resend confirmation." };
  const supabase = await createClient();
  const { error } = await supabase.auth.resend({
    type: "signup",
    email,
    options: { emailRedirectTo: `${appOrigin()}/auth/callback` },
  });
  if (error) return deliveryError(error.code);
  return {
    message: "If confirmation is needed, a new link will arrive in your inbox.",
    verification: true,
  };
}

export async function requestPasswordReset(
  _previous: AuthState,
  form: FormData,
): Promise<AuthState> {
  const { email } = credentials(form);
  if (!validEmail(email)) return { error: "Enter your email address." };
  const supabase = await createClient();
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${appOrigin()}/auth/callback?next=/auth/reset-password`,
  });
  if (error) return deliveryError(error.code);
  return {
    message:
      "If an account exists, a password reset link will arrive in your inbox.",
  };
}

export async function updatePassword(
  _previous: AuthState,
  form: FormData,
): Promise<AuthState> {
  const { password } = credentials(form);
  if (!validPassword(password))
    return { error: "Use a password between 12 and 128 characters." };
  if (password !== form.get("confirmPassword"))
    return { error: "The passwords do not match." };
  const supabase = await createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user)
    return {
      error:
        "Your reset link has expired. Request a new one from the sign-in page.",
    };
  const { error } = await supabase.auth.updateUser({ password });
  if (error)
    return {
      error:
        "Unable to update your password. Request a new reset link and try again.",
    };
  // Force a fresh password login after recovery; revoke other sessions too.
  const { error: logoutError } = await supabase.auth.signOut();
  if (logoutError)
    return {
      error:
        "Password updated, but session cleanup failed. Please sign out before continuing.",
    };
  redirect("/auth/login?notice=password-updated");
}

export async function signInWithGoogle() {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo: `${appOrigin()}/auth/callback`,
      scopes: "openid email profile",
    },
  });
  if (error || !data.url) redirect("/auth/login?error=oauth");
  redirect(data.url);
}

export async function signOut() {
  const supabase = await createClient();
  const { error } = await supabase.auth.signOut({ scope: "local" });
  if (error) redirect("/auth/login?error=signout");
  redirect("/auth/login");
}
