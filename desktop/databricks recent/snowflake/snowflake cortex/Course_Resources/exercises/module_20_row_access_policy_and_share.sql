-- ==========================================================
-- Exercise 20.2 — One Shared View, Three Business Units, Zero Duplicated Views
-- Course: 1018
-- Module 20
-- ==========================================================
--
-- US retail-ops, an international team under stricter data-residency
-- rules, and a smaller foodservice team all need the same packaged
-- forecast app — without three codebases. A row access policy evaluated
-- per row at query time, backed by a role-to-business-unit mapping
-- table, lets one shared view serve all three: the policy compares
-- CURRENT_ROLE() against the mapping table and filters to that BU's own
-- rows. When the international BU's queries came back with zero rows
-- instead of an error, the fix wasn't the share — it was a missing row
-- in the mapping table.
--
-- Your task:
-- 1. Create the row access policy that checks CURRENT_ROLE() against a
--    bu_role_map table and returns TRUE only for that BU's own rows.
-- 2. Attach the policy to the shared sales table's bu_code column.
-- 3. Diagnose: an international-BU query against the shared view returns
--    zero rows with no error. Write the ONE query that tells you whether
--    this is the policy correctly denying, or a genuinely missing share —
--    and identify what table to check first per the diagnosis path.
-- ==========================================================

-- YOUR CODE:

-- Step 1: row access policy — a function evaluated per row at query time
CREATE OR REPLACE ROW ACCESS POLICY bu_scope_policy
  AS (bu_code STRING) RETURNS ___ ->
  EXISTS (
    SELECT 1 FROM bu_role_map
    WHERE bu_role_map.role_name = ___()
      AND bu_role_map.bu_code = bu_code
  );

-- Step 2: attach it to the shared table
ALTER TABLE sales_shared ADD ROW ACCESS POLICY ___ ON (bu_code);

-- Step 3: diagnosis query — confirm this is policy-correct denial, not a
-- broken share. Check as an account admin role first, then check the
-- mapping table for the missing BU row.
SELECT * FROM bu_role_map WHERE role_name = ___;
-- Zero rows here means the policy is CORRECTLY denying everything because
-- the mapping is incomplete, not because the share is broken


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- Step 1: policy checks session context (current role) against a mapping
-- table — 3 BUs, 1 mapping table, 0 duplicated views. The mapping table is
-- the only thing that changes when a 4th BU joins.
-- (conceptual pattern, verify exact clause syntax against current docs)
CREATE OR REPLACE ROW ACCESS POLICY bu_scope_policy
  AS (bu_code STRING) RETURNS BOOLEAN ->
  EXISTS (
    SELECT 1 FROM bu_role_map
    WHERE bu_role_map.role_name = CURRENT_ROLE()
      AND bu_role_map.bu_code = bu_code
  );

-- Step 2: attach the policy to the shared table
ALTER TABLE sales_shared ADD ROW ACCESS POLICY bu_scope_policy ON (bu_code);
-- Test policy changes against a non-production role before applying to
-- the live share.

-- Step 3: diagnosis for "international BU gets zero rows, no error"
-- First check: is this a real access gap or the row access policy
-- correctly filtering rows for that role? Confirm with a query as an
-- account admin role first.
-- Diagnosis path 1: the secure share was created, but the consumer
-- account never ran CREATE DATABASE FROM SHARE — the share exists on the
-- provider side but was never accepted.
-- Diagnosis path 2: the share's underlying database is region-local, and
-- the international BU's account is provisioned in a different region.
-- Root cause found here: bu_role_map has no row mapping the international
-- BU's role to its bu_code — the policy is correctly denying everything
-- because the mapping is incomplete, not because of a broken share.
SELECT * FROM bu_role_map WHERE role_name = 'INTL_BU_ROLE';
-- Fix: INSERT the missing role-to-bu_code row, then re-run the cross-BU
-- access test — attempt an unauthorized cross-BU query and confirm it
-- returns zero rows, not an error that leaks structure. This test becomes
-- the regression check before every future policy edit.
*/
