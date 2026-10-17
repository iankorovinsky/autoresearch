# Authentication

Supabase Auth owns credentials, user identities, OAuth exchange, and email verification. Prisma owns the application tables only. Do not add passwords or a parallel users table to Prisma.

## Current setup

The Auto Research Supabase dashboard was checked on October 16, 2026:

- Email/password sign-in and new signups are enabled.
- Confirm email is enabled; anonymous sign-in is disabled.
- Minimum password length is set to 12 characters, matching the app's server-side checks.
- Site URL is `http://127.0.0.1:3000`.
- Local redirect allowlist contains `http://127.0.0.1:3000/**` and `http://localhost:3000/**`.
- Google OAuth is configured in Supabase and always offered by the frontend.
- Resend custom SMTP is configured with sender `Autoresearch <noreply@iankorovinsky.ca>`. The user confirmed verification email delivery after configuring SMTP.

## Next.js flow

`/auth/login` provides sign-in, signup, confirmation resend, and password reset requests. Server Actions use `@supabase/ssr` to write session cookies. Passwords are passed only to Supabase Auth and are not logged or stored by this application.

`src/proxy.ts` refreshes/validates signed claims and passes refreshed cookies through to the browser. `/` separately calls `getUser()` before rendering the workspace and requires a confirmed email. Session-derived pages are dynamic and proxy responses are private/no-store. Do not authorize requests using the unverified contents of `getSession()`.

The existing default email templates send users through Supabase's verification endpoint and then `/auth/callback`. That route exchanges the PKCE code for a cookie session. Google uses the same callback. Redirect destinations are restricted to `/` and `/auth/reset-password`; arbitrary `next` values are rejected.

Open confirmation/reset links in the **same browser and origin** where the flow started because PKCE depends on the verifier cookie. Use `http://127.0.0.1:3000` consistently with the current `APP_URL`. For cross-browser verification, configure token-hash email templates and a separate verifyOtp handler if cross-browser verification is required; that alternative flow is not implemented here.

The password reset page requires a verified session. Updating the password signs out all Supabase sessions and asks for a fresh sign-in. Normal sign-out ends the current browser session.

Research data is still demo data. Its localStorage keys are scoped by Supabase user ID (`ar:<user-id>:threads`, `:model`, `:gpu`). Legacy global demo keys are not imported into a signed-in account. This avoids accidental sharing in the app but does not encrypt localStorage or protect it from someone with access to the same browser profile. There is no Prisma/database integration yet.

## Environment

Set these in `frontend/.env.local` (see `.env.example`):

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`
- `APP_URL`: canonical app origin used for email and OAuth redirects.

No service-role key is used for authentication. SMTP credentials and Google client secrets belong in Supabase's provider configuration. `data/.env` and its PostgreSQL password are unrelated to user sign-in.

## Google configuration

1. In Google Cloud, configure the consent screen/Google Auth Platform for the app and create a Web application OAuth client.
2. Register the Supabase callback URI shown under Authentication → Sign In / Providers → Google. For this project it is `https://ohmnjlccuifsgpmoeqys.supabase.co/auth/v1/callback` (not the Next.js callback).
3. If the Google app is in testing, add the intended test users. Request only the basic identity scopes (`openid`, `email`, `profile`); do not request Gmail/Drive access.
4. Enter the client ID and secret in Supabase's Google provider panel and enable it. Leave nonce validation enabled and require email.
5. The frontend always offers Google sign-in; there is no feature flag. Use a real user to verify the consent/callback flow.

## Email delivery

Resend sends authentication emails using the verified `iankorovinsky.ca` domain. Supabase SMTP settings use sender name `Autoresearch`, sender address `noreply@iankorovinsky.ca`, host `smtp.resend.com`, port `465`, username `resend`, and the Resend API key as the password. The user configured these settings and confirmed delivery. Credentials stay out of Git and browser bundles.

Keep Confirm email enabled. The default confirmation and reset templates work with the implemented callback; branded/custom templates can be configured in Supabase. Test delivery, resend behavior, expired links, and password recovery with a real account before launch. Do not disable confirmation to work around delivery failures.

Once a production domain exists, change `APP_URL` and Supabase's Site URL to that HTTPS origin and allow its exact auth callback URL. Keep localhost wildcards confined to development. Add rate limiting/CAPTCHA appropriate for a public launch.

## Verification and limits

`npm test` checks redirect restrictions and server input-validation helpers without creating accounts or sending email. `npm run build` checks the full Next.js application. Browser checks cover signed-out redirects, form modes, and the Google sign-in button. Verification email delivery was confirmed by the user. Completing confirmation, session renewal, Google consent, and password recovery still need an end-to-end check with user-entered credentials.

References: [Supabase SSR](https://supabase.com/docs/guides/auth/server-side/creating-a-client), [password authentication](https://supabase.com/docs/guides/auth/passwords), [Google OAuth](https://supabase.com/docs/guides/auth/social-login/auth-google), [SMTP](https://supabase.com/docs/guides/auth/auth-smtp).
