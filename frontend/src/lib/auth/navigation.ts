// Only the recovery flow may select another destination. Never redirect to an
// arbitrary next/redirectTo value supplied by the browser or OAuth provider.
export function authDestination(next: string | null) {
  return next === "/auth/reset-password" ? next : "/";
}
