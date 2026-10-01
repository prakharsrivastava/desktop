# Reconstruction MANIFEST — from 4 screen recordings (2026-06-11)

Source videos (all are over-the-shoulder Zoom recordings of "AM70582"'s screen):

| Video | Length | Shows |
|---|---|---|
| 2026-06-11_09-20-12.mp4 | ~16.5 min | Architecture diagram (Excalidraw) + **e360-id-card-agent** (Bitbucket file listing) |
| 2026-06-11_12-53-02.mp4 | ~11 min | **e360-address-change-agent** code (VS Code) |
| 2026-06-11_13-04-14.mp4 | ~8.5 min | **e360-address-change-agent** code (VS Code) |
| 2026-06-11_13-12-47.mp4 | ~10 min | README/docs + **e360-orchestrator-agent** / e360-inquiry-categorization |

**Honesty note:** videos only show sampled, scrolled views. Big classes
(e.g. `AddressChangeAgent` is ~400 lines) never appear in full on screen. Files
below are marked HIGH / MEDIUM / LOW / STUB depending on how much was legible.
Anything STUB has its role confirmed from the file explorer but no body captured.

---

## Repo 1 — e360-address-change-agent  (this folder, reconstructed)

| File | Class / fn seen | Confidence |
|---|---|---|
| e360_cloud_connect_python_client/protegrity_file_envelope_protection_service.py | `ProtegrityFileEnvelopeProtectionService` (encrypt/decrypt streams) | **HIGH** (fully legible) |
| state_management/db.py | `OracleDBConnector` (`fetch_with_sql`, `drop_task_rows`, `delete_old_tasks`, `@retry_db_operation`) | **HIGH** for drop/delete, MEDIUM for fetch |
| main.py | `health()`, `process_address_change_request()`, `AGENT_TASK_MAPPING` | **HIGH** (top), MEDIUM (FastAPI wiring) |
| usecases/address_change/agent.py | `AddressChangeAgent(E360UseCaseAgent)` + `set_qmcso_fields()` | MEDIUM (one method window only) |
| usecases/base.py | `E360UseCaseAgent(ABC)` (`log_step`, `requires_human_approval`) | MEDIUM |
| agent_toolbox/id_card.py | `IDCardRequestStatus`, `request_new_id_card`, `fetch_id_card_plan_preference` | MEDIUM (windows) |
| agentic_workflow/workflow_state.py | `WorkflowState(BaseModel)` | LOW (brief glimpse — verify fields) |
| agent_toolbox/address.py | `validate_address`, `_parse_address_to_dict` | STUB (signatures seen) |
| external_apis/edp.py | `call_soa_get_hcid_async`, `string_to_base64`, `get_auth_token`, `should_include_meta_src_envmt` | STUB (signatures seen) |
| external_apis/soa.py | `get_auth_soa_async`, `post_id_card_request`, `update_limited_liability_spi_mem` | STUB (referenced) |
| agent_toolbox/member_entities.py | `convert_matched_entities_to_member_entities` | STUB (referenced) |
| logger.py | `setup_logger(name, req_id)` | STUB (referenced) |
| external_apis/{address_validation,inquirecase,inquiry_categorization,pcp_demo,pega,ssl_config}.py | API clients | STUB (explorer only) |
| state_management/{audit_service,db_manager,migrations,state_models,state_registry,state_repository,state_service}.py | state layer | STUB (explorer only) |
| agent_toolbox/{group_details,member_details,utils}.py, address_tools/utils.py | toolbox | STUB (explorer only) |
| usecases/address_change/{common.py, rules/base.py, rules/address_change.py} | rules | STUB (explorer only) |

Confirmed full file tree of this repo is reproduced 1:1 in this folder.

---

## Repo 2 — e360-id-card-agent  (NOT yet reconstructed)

Seen only as a Bitbucket file listing in 2026-06-11_09-20-12.mp4. Folders:
`agent_toolbox/`, `agentic_experts/`, `external_apis/`, `usecases/`,
`state_management/`, plus `main.py`, `Dockerfile`, `requirements.txt`,
`HEALTH_CHECK_CONFIGURATION.md`, `README.md`, `logger.py`, `SSL_SETUP.md`.
Almost no code bodies were shown — needs dense frame reads to populate.

## Repo 3 — e360-orchestrator-agent  (NOT yet reconstructed)

Appears in the "Recent Folders" list and briefly in 2026-06-11_13-12-47.mp4.
This is the **Planner/Orchestrator** (the architecture diagram's "Planner agent"
that routes to executor agents: ID card / Address change / Enquiry). Its
`planning_agent.py` was referenced but not captured at a legible zoom in the
sampled frames. Needs targeted reading.

## Repo 4 — e360-inquiry-categorization  (already in your earlier zip)

Shown again here (`AI/` folder: AI_utils, bundling_utils, category, EA,
filename_utils, filenet, inference, process_bundling, prompts, soa_integration,
summary, validation, work_type_detail, worktype, AI_Agents/, DB/, jobs/).
You already have this project — not re-reconstructed.

---

## How to complete any file

Tell me the repo + file (e.g. "orchestrator planning_agent.py" or
"address-change-agent external_apis/edp.py") and I will do a dense, frame-by-frame
read of that file's segment in the relevant video and fill in the faithful body.
