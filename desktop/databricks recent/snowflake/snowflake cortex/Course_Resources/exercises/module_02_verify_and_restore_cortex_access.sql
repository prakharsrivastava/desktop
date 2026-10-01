-- ==========================================================
-- Exercise 2.1 — Verify and Restore Cortex Access for a Role
-- Course: 1018
-- Module 02
-- ==========================================================
--
-- ACCOUNTADMIN can run Cortex functions out of the box on a fresh account,
-- which is exactly why this is a trap: it hides the fact that ordinary
-- analytics roles need the SNOWFLAKE.CORTEX_USER database role AND the
-- account-level USE AI FUNCTIONS privilege before they can call AI_COMPLETE
-- or any task function. This is a shorter, guided exercise — the module is
-- about access readiness, not writing new SQL logic.
--
-- Your task:
-- 1. Check whether analytics_role still has its default Cortex grant.
-- 2. If governance has revoked SNOWFLAKE.CORTEX_USER, re-grant it.
-- 3. If the account-level USE AI FUNCTIONS privilege was also revoked,
--    re-grant it to the same role.
-- ==========================================================

-- YOUR CODE:

-- 1. Check whether the default PUBLIC grant is still present
SHOW GRANTS TO ROLE ___;

-- 2. If governance revoked it, re-grant the Cortex usage privilege
GRANT DATABASE ROLE SNOWFLAKE.___ TO ROLE ___;

-- 3. If account-level USE AI FUNCTIONS was also revoked, re-grant it too
GRANT ___ ON ACCOUNT TO ROLE analytics_role;


-- ==========================================================
-- SOLUTION (attempt first before scrolling!)
-- ==========================================================
/*
-- 1. Check whether the default PUBLIC grant is still present
SHOW GRANTS TO ROLE analytics_role;

-- 2. If governance revoked it, re-grant the Cortex usage privilege
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE analytics_role;

-- 3. If account-level USE AI FUNCTIONS was also revoked, re-grant it too
GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE analytics_role;
*/
