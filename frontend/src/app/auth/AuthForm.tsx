"use client";

import { useActionState, useState } from "react";
import {
  ArrowRight,
  FlaskConical,
  LoaderCircle,
  Mail,
  ShieldCheck,
} from "lucide-react";
import {
  signIn,
  signUp,
  resendConfirmation,
  requestPasswordReset,
  updatePassword,
  signInWithGoogle,
} from "./actions";
import type { AuthState } from "./actions";
import "./auth.css";

type Mode = "signin" | "signup" | "forgot" | "resend" | "reset";
const copy: Record<
  Mode,
  { title: string; description: string; button: string }
> = {
  signin: {
    title: "Welcome back.",
    description: "A little curiosity goes a long way.",
    button: "Sign in",
  },
  signup: {
    title: "A place for your next discovery.",
    description: "Create an account to start your research.",
    button: "Create account",
  },
  forgot: {
    title: "Let’s get you back in.",
    description: "We’ll send a link to reset your password.",
    button: "Send reset link",
  },
  resend: {
    title: "Check your inbox.",
    description: "Confirm your email to open your workspace.",
    button: "Resend confirmation",
  },
  reset: {
    title: "A fresh start.",
    description: "Choose a new password for your account.",
    button: "Update password",
  },
};
const actions = {
  signin: signIn,
  signup: signUp,
  forgot: requestPasswordReset,
  resend: resendConfirmation,
  reset: updatePassword,
};

export default function AuthForm({
  resetPassword = false,
  initialError,
  initialMessage,
}: {
  resetPassword?: boolean;
  initialError?: string;
  initialMessage?: string;
}) {
  const [mode, setMode] = useState<Mode>(resetPassword ? "reset" : "signin");
  return (
    <main className="auth-page">
      <header className="auth-brand">
        <span className="brand-mark" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span>
          auto<span>research</span>
        </span>
      </header>
      <section className="auth-layout">
        <aside className="auth-story">
          <span className="auth-eyebrow">A WORKSPACE FOR DISCOVERY</span>
          <h1>
            Good questions.
            <br />
            <span>Better experiments.</span>
          </h1>
          <p>
            From an idea to a faster kernel.
            <br />
            Keep your research, reasoning, and results together.
          </p>
          <div className="auth-story-note">
            <FlaskConical size={18} />
            <span>
              Built for the work between
              <br />
              “what if” and “it works.”
            </span>
          </div>
          <div className="auth-orbit" aria-hidden="true">
            <i />
            <i />
            <i />
            <span>
              <FlaskConical size={31} strokeWidth={1} />
            </span>
          </div>
        </aside>
        <section className="auth-card">
          <Form
            key={mode}
            mode={mode}
            setMode={setMode}
            initialError={initialError}
            initialMessage={initialMessage}
          />
        </section>
      </section>
      <footer className="auth-footer">
        <ShieldCheck size={13} /> Secure sign-in, powered by Supabase
      </footer>
    </main>
  );
}
function Form({
  mode,
  setMode,
  initialError,
  initialMessage,
}: {
  mode: Mode;
  setMode: (mode: Mode) => void;
  initialError?: string;
  initialMessage?: string;
}) {
  const [state, formAction, pending] = useActionState<AuthState, FormData>(
    actions[mode],
    {},
  );
  const [email, setEmail] = useState("");
  const details = copy[mode];
  const needsPassword =
    mode === "signin" || mode === "signup" || mode === "reset";
  return (
    <>
      <span className="auth-card-eyebrow">YOUR RESEARCH STARTS HERE</span>
      <h2>{details.title}</h2>
      <p className="auth-description">{details.description}</p>
      {(mode === "signin" || mode === "signup") && (
        <>
          <form action={signInWithGoogle}>
            <button className="google-button" disabled={pending}>
              <span className="google-letter" aria-hidden="true">
                G
              </span>
              Continue with Google
            </button>
          </form>
          <div className="auth-divider">
            <span />
            or continue with email
            <span />
          </div>
        </>
      )}
      <form action={formAction} className="auth-form">
        {mode !== "reset" && (
          <label>
            Email address
            <input
              autoComplete="email"
              type="email"
              name="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              maxLength={254}
              required
              disabled={pending}
            />
          </label>
        )}
        {needsPassword && (
          <div className="auth-field">
            <div className="auth-label-row">
              <label htmlFor="auth-password">
                {mode === "reset" ? "New password" : "Password"}
              </label>
              {mode === "signin" && (
                <button
                  type="button"
                  disabled={pending}
                  onClick={() => setMode("forgot")}
                >
                  Forgot password?
                </button>
              )}
            </div>
            <input
              id="auth-password"
              name="password"
              type="password"
              autoComplete={
                mode === "signin" ? "current-password" : "new-password"
              }
              minLength={mode === "signin" ? undefined : 12}
              maxLength={128}
              placeholder={
                mode === "signin" ? "Your password" : "At least 12 characters"
              }
              required
              disabled={pending}
            />
          </div>
        )}
        {mode === "reset" && (
          <label>
            Confirm new password
            <input
              name="confirmPassword"
              type="password"
              autoComplete="new-password"
              minLength={12}
              maxLength={128}
              required
              disabled={pending}
            />
          </label>
        )}
        {(state.error || (!state.message && initialError)) && (
          <div className="auth-alert error" role="alert">
            {state.error || initialError}
          </div>
        )}
        {(state.message || (!state.error && initialMessage)) && (
          <div className="auth-alert success" role="status">
            <Mail size={16} />
            <span>{state.message || initialMessage}</span>
          </div>
        )}
        <button type="submit" className="auth-submit" disabled={pending}>
          {pending ? (
            <>
              <LoaderCircle size={16} className="spin" />
              Please wait…
            </>
          ) : (
            <>
              {details.button}
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </form>
      {state.verification && mode !== "resend" && (
        <button
          className="auth-secondary"
          disabled={pending}
          onClick={() => setMode("resend")}
        >
          Need a new confirmation email?
        </button>
      )}
      <p className="auth-switch">
        {mode === "signin" ? (
          <>
            New to Auto Research?{" "}
            <button disabled={pending} onClick={() => setMode("signup")}>
              Create an account
            </button>
          </>
        ) : (
          <button disabled={pending} onClick={() => setMode("signin")}>
            Back to sign in
          </button>
        )}
      </p>
      {(mode === "signup" || mode === "resend") && (
        <p className="auth-hint">
          Open the email link in the same browser where you started.
        </p>
      )}
    </>
  );
}
