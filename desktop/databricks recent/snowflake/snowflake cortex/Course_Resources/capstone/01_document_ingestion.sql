-- ==========================================================
-- Capstone Layer 1 — Document Ingestion
-- Meridian Foods: vendor contracts + shipment confirmations
-- Pattern from: module 11 (AI_PARSE_DOCUMENT / AI_EXTRACT)
-- ==========================================================
--
-- Meridian has 1,400+ vendor contracts and shipment confirmations sitting
-- as PDFs in a stage. This layer doesn't redesign module 11's pipeline —
-- it points that same two-step pattern at Meridian's real stage and schema.
-- One DIRECTORY scan touches all 1,400 documents; no per-file script.
--
-- TODO 1: point AI_PARSE_DOCUMENT at Meridian's real vendor stage,
--         in LAYOUT mode, over every file the DIRECTORY() scan returns.
CREATE OR REPLACE TABLE vendor_contracts_parsed AS
SELECT
    relative_path AS doc_name,
    AI_PARSE_DOCUMENT(
        TO_FILE('@___', relative_path),   -- TODO: meridian_vendor_stage
        {'mode': '___'}                    -- TODO: 'LAYOUT'
    ) AS parsed_doc
FROM DIRECTORY(@___);                      -- TODO: same stage as above

-- TODO 2: extract the fields the field-rep assistant (Layer 2) and the
--         anomaly model (Layer 4) will need. lead_time_days feeds the
--         anomaly model directly — don't drop it from the field list.
CREATE OR REPLACE TABLE vendor_terms AS
SELECT
    doc_name,
    AI_EXTRACT(
        parsed_doc:content::STRING,
        [___, ___, ___, ___]   -- TODO: 'vendor_name','lead_time_days','payment_terms','sku_list'
    ) AS terms
FROM vendor_contracts_parsed;

-- TODO 3 (your extension): Meridian also runs promotional flyers through
-- the same stage pattern (see Exercise 22.1). Add a second parse/extract
-- pair here for @meridian_promo_stage, extracting
-- ['sku_list','discount_pct','promo_start','promo_end'] into
-- promo_terms, following the exact same two-statement shape above.

-- Checkpoint 1 (README rubric): vendor_contracts_parsed and vendor_terms
-- populate from a real stage via one DIRECTORY() scan — no per-file script.
