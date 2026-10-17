import "server-only";

export function appOrigin() {
  const configured = process.env.APP_URL;
  if (!configured)
    throw new Error("Set APP_URL to the canonical application URL.");
  const url = new URL(configured);
  const local = url.hostname === "localhost" || url.hostname === "127.0.0.1";
  if (url.protocol !== "https:" && !(local && url.protocol === "http:")) {
    throw new Error("APP_URL must use HTTPS, except for local development.");
  }
  if (url.username || url.password)
    throw new Error("APP_URL must not contain credentials.");
  return url.origin;
}
