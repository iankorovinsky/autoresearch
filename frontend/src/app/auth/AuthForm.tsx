"use client";

import { useActionState, useState } from "react";
import {
  ArrowRight,
  FlaskConical,
  Check,
  Terminal,
  Code2,
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
    title: "Welcome back",
    description: "Sign in to your research workspace.",
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
      <section className="auth-layout">
        <aside className="auth-story">
          <header className="auth-brand">
            <span className="brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span>
              Autoresearch<span className="auth-brand-period">.</span>
            </span>
          </header>
          <div className="auth-story-content">
            <span className="auth-eyebrow">
              <span />
              THE RESEARCH WORKSPACE
            </span>
            <h1>
              From hypothesis
              <br />
              to <span>working code.</span>
            </h1>
            <p>
              A focused space to explore ideas, run experiments,
              <br className="auth-desktop-break" /> and follow the evidence.
            </p>
            <div className="auth-workspace-preview" aria-hidden="true">
              <div className="preview-window-bar">
                <span className="preview-window-dots">
                  <i />
                  <i />
                  <i />
                </span>
                <span>
                  <Terminal size={12} /> kernel-optimization
                </span>
                <span className="preview-demo">PREVIEW</span>
              </div>
              <div className="preview-tabs">
                <span>
                  <Code2 size={12} /> Research
                </span>
                <span>Experiments</span>
                <span>Artifacts</span>
              </div>
              <div className="preview-body">
                <div className="preview-prompt">
                  <span>❯</span> Optimize matrix multiplication on T4
                </div>
                <div className="preview-log">
                  <span>
                    <Check size={12} />
                  </span>{" "}
                  Baseline established <i>2.99 ms</i>
                </div>
                <div className="preview-log">
                  <span>
                    <Check size={12} />
                  </span>{" "}
                  Candidates evaluated <i>12 / 12</i>
                </div>
                <div className="preview-result">
                  <div>
                    <span>BEST CANDIDATE</span>
                    <strong>
                      2.41<span>×</span>
                      <small>faster</small>
                    </strong>
                  </div>
                  <svg viewBox="0 0 210 68">
                    <path d="M4 6L32 12L50 27L75 25L99 40L119 35L144 49L169 53L205 61" />
                    <circle cx="205" cy="61" r="3" />
                  </svg>
                  <span className="preview-result-status">
                    <Check size={11} /> Verified
                  </span>
                </div>
                <div className="preview-input">
                  <span>What should we explore next?</span>
                  <ArrowRight size={13} />
                </div>
              </div>
            </div>
          </div>
          <div className="auth-story-bottom">
            <span>
              <FlaskConical size={14} /> Built for the next iteration.
            </span>
            <span>01 — ∞</span>
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
          <footer className="auth-footer">
            <ShieldCheck size={13} /> Secure sign-in with Supabase
          </footer>
        </section>
      </section>
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
      <span className="auth-card-eyebrow">AUTORESEARCH / WORKSPACE</span>
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
            New to Autoresearch?{" "}
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
