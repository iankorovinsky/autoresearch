BEGIN;

-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateEnum
CREATE TYPE "message_role" AS ENUM ('user', 'assistant', 'system', 'tool');

-- CreateTable
CREATE TABLE "projects" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "owner_id" UUID NOT NULL DEFAULT auth.uid(),
    "name" TEXT NOT NULL,
    "description" TEXT,
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "projects_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "threads" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "project_id" UUID NOT NULL,
    "title" TEXT NOT NULL,
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "threads_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "messages" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "thread_id" UUID NOT NULL,
    "position" INTEGER NOT NULL DEFAULT 0,
    "role" "message_role" NOT NULL,
    "content" TEXT NOT NULL DEFAULT '',
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "messages_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "projects_owner_id_created_at_idx" ON "projects"("owner_id", "created_at");

-- CreateIndex
CREATE INDEX "threads_project_id_updated_at_idx" ON "threads"("project_id", "updated_at");

-- CreateIndex
CREATE UNIQUE INDEX "messages_thread_id_position_key" ON "messages"("thread_id", "position");

-- AddForeignKey
ALTER TABLE "threads" ADD CONSTRAINT "threads_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "projects"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "messages" ADD CONSTRAINT "messages_thread_id_fkey" FOREIGN KEY ("thread_id") REFERENCES "threads"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Prisma models the tables above. Constraints, triggers, and Supabase RLS are
-- maintained in migrations because Prisma Schema Language cannot express them.
ALTER TABLE public.projects ADD CONSTRAINT projects_name_not_blank CHECK (length(btrim(name)) > 0);
ALTER TABLE public.threads ADD CONSTRAINT threads_title_not_blank CHECK (length(btrim(title)) > 0);
ALTER TABLE public.messages ADD CONSTRAINT messages_position_positive CHECK (position > 0);
ALTER TABLE public.messages ADD CONSTRAINT messages_metadata_object CHECK (jsonb_typeof(metadata) = 'object');

CREATE FUNCTION public.set_conversation_updated_at()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := clock_timestamp();
  RETURN NEW;
END;
$$;

CREATE TRIGGER projects_updated_at BEFORE UPDATE ON public.projects
FOR EACH ROW EXECUTE FUNCTION public.set_conversation_updated_at();
CREATE TRIGGER threads_updated_at BEFORE UPDATE ON public.threads
FOR EACH ROW EXECUTE FUNCTION public.set_conversation_updated_at();
CREATE TRIGGER messages_updated_at BEFORE UPDATE ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.set_conversation_updated_at();

-- At PostgreSQL's default READ COMMITTED isolation, the row lock serializes
-- inserts in one thread. The MAX query runs after the previous writer commits.
-- Different threads can accept inserts independently. Never compute MAX in JS.
CREATE FUNCTION public.assign_message_position()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  PERFORM 1 FROM public.threads WHERE id = NEW.thread_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Thread is missing or not accessible' USING ERRCODE = '23503';
  END IF;

  SELECT COALESCE(MAX(position), 0) + 1 INTO NEW.position
  FROM public.messages WHERE thread_id = NEW.thread_id;

  UPDATE public.threads SET updated_at = clock_timestamp() WHERE id = NEW.thread_id;
  RETURN NEW;
END;
$$;

CREATE TRIGGER messages_assign_position BEFORE INSERT ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.assign_message_position();

CREATE FUNCTION public.protect_message_order()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
  IF NEW.thread_id IS DISTINCT FROM OLD.thread_id OR NEW.position IS DISTINCT FROM OLD.position THEN
    RAISE EXCEPTION 'Message thread and position are immutable' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER messages_protect_order BEFORE UPDATE OF thread_id, position ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.protect_message_order();

-- Each project belongs to a Supabase Auth user. auth.users stays managed by
-- Supabase; owner_id is intentionally a scalar UUID rather than a Prisma model.
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.projects, public.threads, public.messages FROM PUBLIC, anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.projects, public.threads, public.messages TO authenticated;
GRANT USAGE ON TYPE public.message_role TO authenticated;

CREATE POLICY projects_owner ON public.projects
FOR ALL TO authenticated
USING (owner_id = (SELECT auth.uid()))
WITH CHECK (owner_id = (SELECT auth.uid()));

CREATE POLICY threads_project_owner ON public.threads
FOR ALL TO authenticated
USING (EXISTS (
  SELECT 1 FROM public.projects p
  WHERE p.id = threads.project_id AND p.owner_id = (SELECT auth.uid())
))
WITH CHECK (EXISTS (
  SELECT 1 FROM public.projects p
  WHERE p.id = threads.project_id AND p.owner_id = (SELECT auth.uid())
));

CREATE POLICY messages_project_owner ON public.messages
FOR ALL TO authenticated
USING (EXISTS (
  SELECT 1 FROM public.threads t JOIN public.projects p ON p.id = t.project_id
  WHERE t.id = messages.thread_id AND p.owner_id = (SELECT auth.uid())
))
WITH CHECK (EXISTS (
  SELECT 1 FROM public.threads t JOIN public.projects p ON p.id = t.project_id
  WHERE t.id = messages.thread_id AND p.owner_id = (SELECT auth.uid())
));

COMMIT;
