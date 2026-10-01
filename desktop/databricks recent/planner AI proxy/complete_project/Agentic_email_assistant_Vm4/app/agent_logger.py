"""
Agent Step Logger Module
Provides persistent step-level logging for email intelligence agent.
Supports SQLite (default) with easy migration to PostgreSQL/SQL Server.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager
import os


class AgentLogger:
    """
    Persistent logger for tracking agent processing steps.
    Database-agnostic design for easy migration.
    """

    def __init__(self, db_path: str = "agent_logs.db", db_type: str = "sqlite"):
        """
        Initialize the logger.

        Args:
            db_path: Path to database file (SQLite) or connection string (PostgreSQL/SQL Server)
            db_type: Database type - "sqlite", "postgresql", or "sqlserver"
        """
        self.db_path = db_path
        self.db_type = db_type
        self._init_database()

    def _init_database(self):
        """Create the logs table if it doesn't exist."""
        if self.db_type == "sqlite":
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        input_data TEXT,
                        output_data TEXT,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        duration_ms INTEGER,
                        user_id TEXT
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON agent_logs(request_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_logs(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_name ON agent_logs(step_name)")
                conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection (context manager for automatic cleanup)."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def log_step(
        self,
        request_id: str,
        step_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log a processing step.

        Args:
            request_id: Unique identifier for the request
            step_name: Name of the processing step
            input_data: Input data for this step (will be JSON serialized)
            output_data: Output data from this step (will be JSON serialized)
            status: Status of the step ("success", "error", "warning")
            error_message: Error message if status is "error"
            duration_ms: Duration of the step in milliseconds
            user_id: User who initiated the request
        """
        timestamp = datetime.utcnow().isoformat()

        # Serialize data to JSON
        input_json = json.dumps(input_data) if input_data else None
        output_json = json.dumps(output_data) if output_data else None

        if self.db_type == "sqlite":
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO agent_logs
                    (request_id, step_name, timestamp, input_data, output_data,
                     status, error_message, duration_ms, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_id, step_name, timestamp, input_json, output_json,
                    status, error_message, duration_ms, user_id
                ))
                conn.commit()

    def get_logs(
        self,
        request_id: Optional[str] = None,
        step_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        """
        Retrieve logs with optional filtering.

        Args:
            request_id: Filter by request ID
            step_name: Filter by step name
            status: Filter by status
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of log records as dictionaries
        """
        query = "SELECT * FROM agent_logs WHERE 1=1"
        params = []

        if request_id:
            query += " AND request_id = ?"
            params.append(request_id)

        if step_name:
            query += " AND step_name = ?"
            params.append(step_name)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        if self.db_type == "sqlite":
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                return [dict(row) for row in rows]

        return []

    def get_request_logs(self, request_id: str) -> list:
        """Get all logs for a specific request ID."""
        return self.get_logs(request_id=request_id, limit=1000)


# Global logger instance
_logger_instance: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentLogger()
    return _logger_instance


def log_step(
    request_id: str,
    step_name: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Convenience function to log a step using the global logger.

    Args:
        request_id: Unique identifier for the request
        step_name: Name of the processing step
        input_data: Input data for this step
        output_data: Output data from this step
        status: Status of the step ("success", "error", "warning")
        error_message: Error message if status is "error"
        duration_ms: Duration of the step in milliseconds
        user_id: User who initiated the request
    """
    logger = get_logger()
    logger.log_step(
        request_id=request_id,
        step_name=step_name,
        input_data=input_data,
        output_data=output_data,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
        user_id=user_id
    )
