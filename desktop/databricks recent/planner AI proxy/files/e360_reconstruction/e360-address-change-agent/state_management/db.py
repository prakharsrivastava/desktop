"""
state_management.db

Reconstructed from screen recording 2026-06-11_13-04-14.mp4.
Confidence: HIGH for drop_task_rows / delete_old_tasks (lines fully visible);
MEDIUM for fetch_with_sql (only the tail / except block was on screen);
class header `class OracleDBConnector:` confirmed.

Lines above ~290 (session setup, __init__, get_session, engine creation) were
scrolled past in the recording and are reconstructed as faithful stubs below.
"""

from datetime import datetime

# NOTE: `Task`, `retry_db_operation`, and `logger` are referenced but their
# definitions/imports were off-screen. Faithful placeholders are provided so the
# module is importable; wire to the real ORM model / decorator when available.
try:  # pragma: no cover - depends on the real state_models module
    from state_management.state_models import Task
except Exception:  # placeholder
    Task = None


def retry_db_operation(func):
    """Decorator seen as @retry_db_operation on delete_old_tasks.

    The real implementation (retry/backoff on transient DB errors) was not shown
    in the recording. This passthrough keeps behaviour intact until restored.
    """
    return func


class OracleDBConnector:
    # --- not shown in recording (scrolled past): -------------------------
    #   __init__(self, ...): engine / sessionmaker setup
    #   get_session(self): contextmanager yielding a SQLAlchemy session
    #   create_tables(self) / health-check helpers
    # ---------------------------------------------------------------------

    def fetch_with_sql(self, *args, **kwargs):
        # Only the tail of this method was legible in the recording:
        try:
            raise NotImplementedError(
                "fetch_with_sql body not fully captured in recording"
            )
        except Exception as e:
            logger.error(f"SQL fetch error: {e}")  # noqa: F821 - logger off-screen
            raise

    def drop_task_rows(self, request_id: str) -> int:
        logger.info(f"Deleting state for request {request_id}")  # noqa: F821
        try:
            with self.get_session() as session:
                deleted_count = (
                    session.query(Task).filter_by(request_id=request_id).delete()
                )
                logger.info(  # noqa: F821
                    f"Deleted {deleted_count} tasks for request {request_id}"
                )
                return deleted_count
        except Exception as e:
            logger.error(  # noqa: F821
                f"Error deleting tasks for request {request_id}: {e}"
            )
            raise

    @retry_db_operation
    def delete_old_tasks(self, cutoff_time: datetime) -> int:
        """Delete tasks older than cutoff time"""
        try:
            with self.get_session() as session:
                return (
                    session.query(Task)
                    .filter(Task.updated_at < cutoff_time)
                    .delete()
                )
        except Exception as e:
            logger.error(f"Error deleting old tasks: {e}")  # noqa: F821
            raise
