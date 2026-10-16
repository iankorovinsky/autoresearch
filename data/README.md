# Conversation data

Prisma manages three application tables in Supabase PostgreSQL:

- `projects`: owner UUID, name, optional description, and timestamps.
- `threads`: project, title, and timestamps.
- `messages`: thread, database-assigned position, role (`user`, `assistant`, `system`, or `tool`), text content, JSON metadata, and timestamps.

No research-run, candidate, or evaluation tables are included. There is no seed script or frontend integration.

## Validate locally

Use Node 24 (`.nvmrc` is included):

```sh
cd data
nvm use
npm ci
npm run schema:validate
npm test
```

These checks need no database URL. Tests execute the migration in disposable, in-memory PostgreSQL through PGlite and cover ordering, streaming updates, row-level security, invalid input, and cascade deletion. Their temporary fixtures never connect to Supabase. PGlite does not simulate multiple concurrent PostgreSQL connections.

## Create the tables in Supabase later

Copy `.env.example` to `data/.env`. Set `DATABASE_URL` to the PostgreSQL connection string from Supabase's Connect dialog, including the database password. Use the direct connection or session pooler on port 5432 if the direct IPv6 address is unreachable. URL-encode password characters. The publishable API key is not a PostgreSQL password.

Then, from `data/`, run:

```sh
npm run migrate:deploy
npm run migrate:status
```

The checked-in migration creates the three application tables and records Prisma migration metadata. It inserts no application rows. Do not substitute `db push`: Prisma's schema alone does not include custom checks, ordering triggers, or access policies.

`data/.env` is ignored. The database URL is server-only and must not go in `NEXT_PUBLIC_*` or frontend code. No Supabase migration has been applied during this preparation.

## Ordering and streaming

An insert trigger locks the parent thread row, assigns `MAX(position) + 1`, and updates the thread's activity timestamp. The lock serializes competing inserts into one thread at PostgreSQL's default **READ COMMITTED** isolation. Different threads do not share a lock. Callers using stricter isolation levels must retry serialization/uniqueness failures.

Callers **omit `position`**. The Prisma default of `0` is only an insert placeholder; the trigger replaces it before the positive-position check and unique `(thread_id, position)` constraint run. Supplied positions are ignored. Existing messages cannot change thread or position.

Read messages with `ORDER BY position ASC`. For streaming, create one empty assistant message and update its content. Its position stays fixed. Deletions can leave gaps; deleting the final message allows that final position to be reused. Position is display order, not a permanent event sequence or synchronization cursor.

## Ownership and lifecycle

All three tables have Supabase row-level security. Authenticated users can access only their own projects and their threads/messages. Anonymous access is revoked. Projects default `owner_id` to `auth.uid()`; privileged backend connections must supply the real owner's UUID explicitly because they lack user JWT context and may bypass RLS.

Supabase manages `auth.users`; Prisma does not migrate the auth schema. `owner_id` is a scalar UUID without a foreign key into that external schema, so account deletion cleanup must be implemented with the account lifecycle. Project deletion cascades to threads and messages. Thread deletion cascades to messages.

Models, relations, UUID defaults, indexes, timestamps, and the role enum live in `prisma/schema.prisma`. Custom checks, update triggers, message ordering, and RLS live in the SQL migration. Trigger functions run as the caller, not as a privileged definer.

## Future schema changes

```sh
npm run schema:format
npm run schema:validate
npm run migrate:dev -- --name describe_change --create-only
# Review SQL and add any needed trigger/policy changes.
npm run migrate:dev
```

Use `migrate:dev` only with a disposable development database. It may require a separate shadow database and can request a reset on drift. Never accept a reset against the shared Supabase database. Use `migrate:deploy` for reviewed migrations there.

The initial table SQL was generated with Prisma 7.10.1's schema diff, then extended with the SQL Prisma cannot represent. No runtime Prisma Client is generated yet: this package manages schemas and migrations only.
