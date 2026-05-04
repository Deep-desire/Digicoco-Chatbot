-- Supabase schema for lead capture and chat history
-- Run this in the Supabase SQL Editor.

create extension if not exists pgcrypto;

create table if not exists public.users (
  email text primary key,
  name text not null default '',
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now()
);

create or replace function public.set_users_last_seen_at()
returns trigger
language plpgsql
as $$
begin
  new.last_seen_at := now();
  if new.first_seen_at is null then
    new.first_seen_at := now();
  end if;
  return new;
end;
$$;

drop trigger if exists trg_users_last_seen_at on public.users;
create trigger trg_users_last_seen_at
before insert or update on public.users
for each row
execute function public.set_users_last_seen_at();

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id text not null,
  email text not null references public.users(email) on delete cascade,
  name text not null default '',
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  "timestamp" timestamptz not null default now()
);

create index if not exists idx_messages_email on public.messages (email);
create index if not exists idx_messages_session_id on public.messages (session_id);
create index if not exists idx_messages_timestamp on public.messages ("timestamp" desc);

-- Sessions table: maps a session identifier to a user (email/name) and tracks activity
create table if not exists public.sessions (
  session_id text primary key,
  email text not null references public.users(email) on delete cascade,
  name text not null default '',
  created_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now()
);

create index if not exists idx_sessions_email on public.sessions (email);
create index if not exists idx_sessions_last_seen on public.sessions (last_seen_at desc);

-- Optional: enable Row Level Security if you want client-side access policies.
-- For a backend-only setup using the Supabase service role key, RLS can stay disabled.
-- alter table public.users enable row level security;
-- alter table public.messages enable row level security;
