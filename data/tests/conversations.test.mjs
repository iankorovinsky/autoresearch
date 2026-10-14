import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { after, before, beforeEach, test } from "node:test";
import { PGlite } from "@electric-sql/pglite";

// All fixtures live in an in-memory PostgreSQL instance, never in Supabase.
const db = new PGlite();
const owner = "11111111-1111-4111-8111-111111111111";
const outsider = "22222222-2222-4222-8222-222222222222";
const migration = new URL(
  "../prisma/migrations/20261016233000_create_conversations/migration.sql",
  import.meta.url,
);

before(async () => {
  // Model only the Supabase auth boundary needed by the application migration.
  await db.exec(`
    CREATE ROLE anon NOLOGIN;
    CREATE ROLE authenticated NOLOGIN;
    CREATE SCHEMA auth;
    CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
      $$ SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
    GRANT USAGE ON SCHEMA public, auth TO authenticated;
  `);
  await db.exec(await readFile(migration, "utf8"));
});
beforeEach(async () => {
  await db.exec("TRUNCATE public.projects CASCADE");
});
after(async () => {
  await db.close();
});

async function asUser(userId, callback) {
  return db.transaction(async (tx) => {
    await tx.exec("SET LOCAL ROLE authenticated");
    await tx.query("SELECT set_config('request.jwt.claim.sub', $1, true)", [
      userId,
    ]);
    return callback(tx);
  });
}
async function createThread() {
  return asUser(owner, async (tx) => {
    const {
      rows: [project],
    } = await tx.query(
      "INSERT INTO projects (name) VALUES ('Test project') RETURNING *",
    );
    const {
      rows: [thread],
    } = await tx.query(
      "INSERT INTO threads (project_id, title) VALUES ($1, 'Test thread') RETURNING *",
      [project.id],
    );
    return { project, thread };
  });
}

test("migration creates exactly three empty application tables", async () => {
  const { rows } = await db.query(
    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename",
  );
  assert.deepEqual(
    rows.map((r) => r.tablename),
    ["messages", "projects", "threads"],
  );
  for (const table of ["projects", "threads", "messages"]) {
    const result = await db.query(
      `SELECT count(*)::int AS count FROM ${table}`,
    );
    assert.equal(result.rows[0].count, 0);
  }
});

test("positions are database-assigned per thread and streaming updates preserve order", async () => {
  const { thread } = await createThread();
  await asUser(owner, async (tx) => {
    const {
      rows: [first],
    } = await tx.query(
      "INSERT INTO messages (thread_id, role, content, position) VALUES ($1, 'user', 'Question', 999) RETURNING *",
      [thread.id],
    );
    const {
      rows: [second],
    } = await tx.query(
      "INSERT INTO messages (thread_id, role) VALUES ($1, 'assistant') RETURNING *",
      [thread.id],
    );
    assert.equal(first.position, 1);
    assert.equal(second.position, 2);
    assert.equal(second.content, "");
    const updated = await tx.query(
      "UPDATE messages SET content = 'Streaming answer' WHERE id = $1 RETURNING *",
      [second.id],
    );
    assert.equal(updated.rows[0].position, 2);
    assert.ok(
      new Date(updated.rows[0].updated_at) >= new Date(second.updated_at),
    );
    const {
      rows: [other],
    } = await tx.query(
      "INSERT INTO threads (project_id, title) VALUES ($1, 'Other thread') RETURNING id",
      [thread.project_id],
    );
    const independent = await tx.query(
      "INSERT INTO messages (thread_id, role) VALUES ($1, 'user') RETURNING position",
      [other.id],
    );
    assert.equal(independent.rows[0].position, 1);
  });
});

test("message order cannot be reassigned after insertion", async () => {
  const { thread } = await createThread();
  const message = await asUser(
    owner,
    async (tx) =>
      (
        await tx.query(
          "INSERT INTO messages (thread_id, role) VALUES ($1, 'user') RETURNING id",
          [thread.id],
        )
      ).rows[0],
  );
  await assert.rejects(
    asUser(owner, (tx) =>
      tx.query("UPDATE messages SET position = 10 WHERE id = $1", [message.id]),
    ),
    /immutable/,
  );
  await assert.rejects(
    asUser(owner, (tx) =>
      tx.query(
        "UPDATE messages SET thread_id = gen_random_uuid() WHERE id = $1",
        [message.id],
      ),
    ),
    /immutable/,
  );
});

test("RLS isolates projects, threads, and messages between owners", async () => {
  const { project, thread } = await createThread();
  await asUser(owner, (tx) =>
    tx.query(
      "INSERT INTO messages (thread_id, role, content) VALUES ($1, 'user', 'Private')",
      [thread.id],
    ),
  );
  await asUser(outsider, async (tx) => {
    for (const table of ["projects", "threads", "messages"]) {
      assert.deepEqual((await tx.query(`SELECT * FROM ${table}`)).rows, []);
    }
    assert.equal(
      (
        await tx.query(
          "UPDATE projects SET name = 'Changed' WHERE id = $1 RETURNING id",
          [project.id],
        )
      ).rows.length,
      0,
    );
    assert.equal(
      (
        await tx.query("DELETE FROM threads WHERE id = $1 RETURNING id", [
          thread.id,
        ])
      ).rows.length,
      0,
    );
  });
  await assert.rejects(
    asUser(outsider, (tx) =>
      tx.query("INSERT INTO messages (thread_id, role) VALUES ($1, 'user')", [
        thread.id,
      ]),
    ),
    /not accessible|row-level security/,
  );
  await assert.rejects(
    asUser(outsider, (tx) =>
      tx.query(
        "INSERT INTO threads (project_id, title) VALUES ($1, 'Intrusion')",
        [project.id],
      ),
    ),
    /row-level security/,
  );
  await assert.rejects(
    asUser(outsider, (tx) =>
      tx.query(
        "INSERT INTO projects (owner_id, name) VALUES ($1, 'Forged owner')",
        [owner],
      ),
    ),
    /row-level security/,
  );
});

test("anonymous requests cannot read or write application tables", async () => {
  await assert.rejects(
    db.transaction(async (tx) => {
      await tx.exec("SET LOCAL ROLE anon");
      await tx.query("SELECT * FROM projects");
    }),
    /permission denied/,
  );
});

test("foreign keys, roles, and metadata reject invalid messages", async () => {
  const { thread } = await createThread();
  await assert.rejects(
    db.query(
      "INSERT INTO messages (thread_id, role) VALUES (gen_random_uuid(), 'user')",
    ),
    /missing or not accessible/,
  );
  await assert.rejects(
    db.query("INSERT INTO messages (thread_id, role) VALUES ($1, 'invalid')", [
      thread.id,
    ]),
    /invalid input value/,
  );
  await assert.rejects(
    db.query(
      "INSERT INTO messages (thread_id, role, metadata) VALUES ($1, 'user', '[]')",
      [thread.id],
    ),
    /messages_metadata_object/,
  );
});

test("project deletion cascades to its threads and messages", async () => {
  const { project, thread } = await createThread();
  await asUser(owner, async (tx) => {
    await tx.query(
      "INSERT INTO messages (thread_id, role) VALUES ($1, 'user')",
      [thread.id],
    );
    await tx.query("DELETE FROM projects WHERE id = $1", [project.id]);
    assert.deepEqual((await tx.query("SELECT * FROM threads")).rows, []);
    assert.deepEqual((await tx.query("SELECT * FROM messages")).rows, []);
  });
});
