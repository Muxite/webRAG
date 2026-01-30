-- Fix RLS policies for profiles table to allow users to create their own profiles
-- This allows the quota system to work with user tokens when service role key is unavailable

-- Enable RLS if not already enabled
ALTER TABLE IF EXISTS profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (to avoid conflicts)
DROP POLICY IF EXISTS "Users can view own profiles" ON profiles;
DROP POLICY IF EXISTS "Users can insert own profiles" ON profiles;
DROP POLICY IF EXISTS "Users can update own profiles" ON profiles;
DROP POLICY IF EXISTS "Service role can manage all profiles" ON profiles;

-- Policy: Users can view their own profile
-- Cast both to TEXT for comparison (handles both UUID and TEXT user_id columns)
CREATE POLICY "Users can view own profiles"
    ON profiles FOR SELECT
    USING (auth.uid()::text = user_id::text);

-- Policy: Users can insert their own profile
CREATE POLICY "Users can insert own profiles"
    ON profiles FOR INSERT
    WITH CHECK (auth.uid()::text = user_id::text);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profiles"
    ON profiles FOR UPDATE
    USING (auth.uid()::text = user_id::text)
    WITH CHECK (auth.uid()::text = user_id::text);

-- Policy: Service role can manage all profiles (for backend operations)
-- This policy allows service role key to bypass RLS when available
CREATE POLICY "Service role can manage all profiles"
    ON profiles FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
