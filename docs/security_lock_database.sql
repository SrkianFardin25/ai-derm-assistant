-- ==============================================================================
-- DermAssist AI — Sovereign Database Security Lockdown Script
-- Run this script in the Supabase SQL Editor to enforce strict RLS lockdown.
-- ==============================================================================

-- 1. Enforce Row Level Security (RLS) on all public tables
ALTER TABLE IF EXISTS public.scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.scans FORCE ROW LEVEL SECURITY;

ALTER TABLE IF EXISTS public.lesion_visits ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.lesion_visits FORCE ROW LEVEL SECURITY;

-- 2. Drop dangerous public update and delete policies
DROP POLICY IF EXISTS "Allow public update to scans" ON public.scans;
DROP POLICY IF EXISTS "Allow public delete from scans" ON public.scans;
DROP POLICY IF EXISTS "Allow public update to lesion_visits" ON public.lesion_visits;
DROP POLICY IF EXISTS "Allow public delete from lesion_visits" ON public.lesion_visits;

-- 3. Verify / Recreate explicit immutable append-only policies
DROP POLICY IF EXISTS "Allow public read access to scans" ON public.scans;
CREATE POLICY "Allow public read access to scans" ON public.scans
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public insert to scans" ON public.scans;
CREATE POLICY "Allow public insert to scans" ON public.scans
    FOR INSERT TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "Allow public read access to lesion_visits" ON public.lesion_visits;
CREATE POLICY "Allow public read access to lesion_visits" ON public.lesion_visits
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public insert to lesion_visits" ON public.lesion_visits;
CREATE POLICY "Allow public insert to lesion_visits" ON public.lesion_visits
    FOR INSERT TO anon, authenticated WITH CHECK (true);

-- 4. Revoke destructive DDL / DML permissions from anon & authenticated roles
REVOKE TRUNCATE, DELETE, UPDATE ON public.scans FROM anon;
REVOKE TRUNCATE, DELETE, UPDATE ON public.lesion_visits FROM anon;

-- 5. Harden storage.objects bucket permissions
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('scans', 'scans', true, 15728640, ARRAY['image/jpeg', 'image/png', 'image/webp'])
ON CONFLICT (id) DO UPDATE SET
    file_size_limit = 15728640,
    allowed_mime_types = ARRAY['image/jpeg', 'image/png', 'image/webp'];

DROP POLICY IF EXISTS "Allow public uploads to scans bucket" ON storage.objects;
CREATE POLICY "Allow public uploads to scans bucket" ON storage.objects
    FOR INSERT TO anon, authenticated WITH CHECK (
        bucket_id = 'scans'
        AND (LOWER(storage.extension(name)) IN ('jpg', 'jpeg', 'png', 'webp'))
    );

DROP POLICY IF EXISTS "Allow public view from scans bucket" ON storage.objects;
CREATE POLICY "Allow public view from scans bucket" ON storage.objects
    FOR SELECT TO anon, authenticated USING (bucket_id = 'scans');

DROP POLICY IF EXISTS "Allow public update to scans bucket" ON storage.objects;
DROP POLICY IF EXISTS "Allow public delete from scans bucket" ON storage.objects;
REVOKE DELETE, UPDATE ON storage.objects FROM anon;

-- 6. Verification query to confirm RLS status
SELECT tablename, rowsecurity, hasindexes
FROM pg_tables
WHERE schemaname = 'public' AND tablename IN ('scans', 'lesion_visits');
