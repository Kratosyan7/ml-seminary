import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class AuditStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    policy_doc_id TEXT,
                    contract_doc_ids_json TEXT,
                    needs_change_count INTEGER,
                    payload_json TEXT,
                    error TEXT
                )
                """
            )
            conn.commit()

    def record_run(
        self,
        *,
        run_id: str,
        created_at: str,
        event_type: str,
        status: str,
        policy_doc_id: Optional[str],
        contract_doc_ids: List[str],
        payload: Optional[Dict[str, Any]],
        needs_change_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_runs (
                    id, created_at, event_type, status, policy_doc_id,
                    contract_doc_ids_json, needs_change_count, payload_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    event_type,
                    status,
                    policy_doc_id,
                    json.dumps(contract_doc_ids, ensure_ascii=False),
                    needs_change_count,
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                    error,
                ),
            )
            conn.commit()

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM audit_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "status": row["status"],
                "policy_doc_id": row["policy_doc_id"],
                "contract_doc_ids": json.loads(row["contract_doc_ids_json"] or "[]"),
                "needs_change_count": row["needs_change_count"],
                "payload": json.loads(row["payload_json"]) if row["payload_json"] else None,
                "error": row["error"],
            })
        return result
