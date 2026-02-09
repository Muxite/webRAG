# Security

## Security Model
- **Gateway access** uses Supabase Auth JWTs only
- **Users** authenticate with Supabase email/password and confirm email before use
- **Per-user limits** are enforced as daily tick quotas stored in Supabase
- **Secrets management** uses env vars locally and AWS Secrets Manager in production

## Supabase Authentication
- Supabase manages `auth.users` and sends confirmation emails on sign-up
- The frontend obtains an access token from Supabase and sends it as `Authorization: Bearer <token>`
- The Gateway validates tokens with `SUPABASE_JWT_SECRET` and rejects invalid or unconfirmed users

### Environment Variables
**Required:**
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_ANON_PUBLIC_KEY` — Publishable/anon key for client auth and RLS
- `SUPABASE_JWT_SECRET` — JWT secret for verifying user tokens

**Server-side only:**
- `SUPABASE_SERVICE_ROLE_KEY` — Service role key for gateway-only Supabase writes

**Optional:**
- `SUPABASE_ALLOW_UNCONFIRMED` — Set to `"true"` to allow unconfirmed emails in development

**Setting environment variables:**
Provide these via `services/.env` and `services/keys.env`. Do not commit secrets.

## Per-User Tick Quotas
- Each user has a profile row in `profiles` with `daily_tick_limit` (default 32)
- Daily usage is tracked in `user_daily_usage` keyed by `(user_id, usage_date)`
- On each `/tasks` call, the Gateway subtracts `max_ticks` and returns `429` when exhausted
- RLS policies enforce user-scoped access for `profiles`, `tasks`, and `user_daily_usage`

## Secrets Management
**Local**: API keys in `services/keys.env`, env vars in `services/.env`, loaded by Docker Compose

**AWS**: Secrets via AWS Secrets Manager, task definitions reference secrets, no secrets in images

## Connection Security
RabbitMQ/Redis use standard protocols locally. External APIs use HTTPS. ChromaDB uses HTTP locally.

For AWS production: VPC isolation, TLS/SSL for inter-service communication, encrypted EBS volumes.
