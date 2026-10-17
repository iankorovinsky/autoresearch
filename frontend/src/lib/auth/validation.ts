export function credentials(form: FormData) {
  return {
    email: String(form.get("email") ?? "").trim(),
    // Whitespace may be intentional in passwords. Do not trim it.
    password: String(form.get("password") ?? ""),
  };
}
export function validEmail(email: string) {
  return email.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
export function validPassword(password: string) {
  return password.length >= 12 && password.length <= 128;
}
