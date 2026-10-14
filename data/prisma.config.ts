import { config } from "dotenv";
import { fileURLToPath } from "node:url";
import { defineConfig } from "prisma/config";

config({ path: fileURLToPath(new URL(".env", import.meta.url)), quiet: true });

export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: { path: "prisma/migrations" },
  // Local schema validation and SQL generation do not need credentials.
  // Migration commands require DATABASE_URL; no fallback database is used.
  ...(process.env.DATABASE_URL
    ? { datasource: { url: process.env.DATABASE_URL } }
    : {}),
});
