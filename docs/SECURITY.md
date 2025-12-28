## Security model

- **Gateway access** is controlled by Supabase Auth JWTs only.
- **Users** authenticate with email and password in Supabase and must confirm their email before they can use the site.
- **Per-user limits** are enforced as daily tick quotas stored in Supabase.

## Supabase authentication

- Supabase manages `auth.users` and sends confirmation emails on sign-up.
- The frontend obtains an access token from Supabase and sends it as `Authorization: Bearer <token>`.
- The Gateway validates the token using `SUPABASE_JWT_SECRET` and rejects requests when the token is invalid or the email is not confirmed.

### Environment variables

See [docs/SUPABASE_SETUP.md](SUPABASE_SETUP.md) for detailed setup instructions.

**Required:**
- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_ANON_KEY` — Publishable/anon key (NOT service role key) - use this for RLS
- `SUPABASE_JWT_SECRET` — JWT secret for verifying user tokens

**Optional:**
- `SUPABASE_ALLOW_UNCONFIRMED` — Set to `"true"` to allow unconfirmed emails in development

**How to get `SUPABASE_JWT_SECRET`:**
1. Go to your Supabase project dashboard: https://supabase.com/dashboard
2. Navigate to **Settings** → **API**
3. Scroll down to the **JWT Settings** section
4. Copy the `JWT_SECRET` value (this is your `SUPABASE_JWT_SECRET`)

**Setting environment variables:**
These values are treated as secrets and should only be provided via environment or Docker compose files, never committed to source control. The gateway container loads environment variables from:
- `.env` file (in the `services/` directory)
- `keys.env` file (in the `services/` directory)

Add `SUPABASE_JWT_SECRET=<your-secret>` to one of these files to make it available to the gateway container.

## Per-user tick quotas

- Each user has a profile row in the `profiles` table with `daily_tick_limit` (default 32).
- Daily usage is tracked in `user_daily_usage` keyed by `(user_id, usage_date)`.
- On each `/tasks` call, the Gateway subtracts `max_ticks` from the user's remaining ticks and rejects the call with `429` when the quota is exhausted.
- **Row-Level Security (RLS):** The `profiles` and `user_daily_usage` tables use RLS policies. See [docs/SUPABASE_SETUP.md](SUPABASE_SETUP.md) for setup instructions.