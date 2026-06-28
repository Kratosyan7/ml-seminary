import json
import os
from contextlib import contextmanager
from datetime import datetime, UTC
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(primary_key=True)
    doc_type: Mapped[str] = mapped_column(index=True)
    filename: Mapped[str] = mapped_column()
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(default=_now_iso)
    updated_at: Mapped[str] = mapped_column(default=_now_iso)


class AuditRunORM(Base):
    __tablename__ = "audit_runs"

    id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[str] = mapped_column(index=True)
    event_type: Mapped[str] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(index=True)
    policy_doc_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    contract_doc_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    needs_change_count: Mapped[Optional[int]] = mapped_column(nullable=True)
    payload_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class BatchJobORM(Base):
    __tablename__ = "batch_jobs"

    job_id: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(index=True)
    created_at: Mapped[str] = mapped_column(index=True)
    completed_at: Mapped[Optional[str]] = mapped_column(nullable=True)
    policy_doc_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    contract_doc_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class SentEmailORM(Base):
    __tablename__ = "sent_emails"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sent_at: Mapped[str] = mapped_column(index=True)
    doc_id: Mapped[str] = mapped_column(index=True)
    recipients_json: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(index=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PersistenceStore:
    def __init__(self, database_url: Optional[str] = None):
        default_sqlite = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audit.db'))}"
        self.database_url = _normalize_database_url(database_url or os.getenv("DATABASE_URL", default_sqlite))
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            future=True,
            expire_on_commit=False,
        )
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ping(self) -> bool:
        try:
            with self.session() as db:
                db.execute(select(1))
            return True
        except Exception:
            return False

    def upsert_document(self, *, doc_id: str, doc_type: str, filename: str, text: str) -> None:
        with self.session() as db:
            row = db.get(DocumentORM, doc_id)
            if row is None:
                row = DocumentORM(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    filename=filename,
                    text=text,
                    created_at=_now_iso(),
                    updated_at=_now_iso(),
                )
                db.add(row)
            else:
                row.doc_type = doc_type
                row.filename = filename
                row.text = text
                row.updated_at = _now_iso()

    def list_documents(self) -> List[Dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(select(DocumentORM)).scalars().all()
        return [
            {"doc_id": row.doc_id, "doc_type": row.doc_type, "filename": row.filename, "text": row.text}
            for row in rows
        ]

    def has_doc(self, doc_id: str, doc_type: Optional[str] = None) -> bool:
        with self.session() as db:
            row = db.get(DocumentORM, doc_id)
        if row is None:
            return False
        if doc_type is not None and row.doc_type != doc_type:
            return False
        return True

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
        with self.session() as db:
            row = db.get(AuditRunORM, run_id)
            if row is None:
                row = AuditRunORM(id=run_id)
                db.add(row)
            row.created_at = created_at
            row.event_type = event_type
            row.status = status
            row.policy_doc_id = policy_doc_id
            row.contract_doc_ids_json = json.dumps(contract_doc_ids, ensure_ascii=False)
            row.needs_change_count = needs_change_count
            row.payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
            row.error = error

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                select(AuditRunORM).order_by(AuditRunORM.created_at.desc()).limit(limit)
            ).scalars().all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at,
                "event_type": row.event_type,
                "status": row.status,
                "policy_doc_id": row.policy_doc_id,
                "contract_doc_ids": json.loads(row.contract_doc_ids_json or "[]"),
                "needs_change_count": row.needs_change_count,
                "payload": json.loads(row.payload_json) if row.payload_json else None,
                "error": row.error,
            }
            for row in rows
        ]

    def create_job(self, *, job_id: str, created_at: str, policy_doc_id: str, contract_doc_ids: List[str]) -> None:
        with self.session() as db:
            row = BatchJobORM(
                job_id=job_id,
                status="queued",
                created_at=created_at,
                completed_at=None,
                policy_doc_id=policy_doc_id,
                contract_doc_ids_json=json.dumps(contract_doc_ids, ensure_ascii=False),
                result_json=None,
                error=None,
            )
            db.merge(row)

    def update_job(
        self,
        *,
        job_id: str,
        status: str,
        completed_at: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.session() as db:
            row = db.get(BatchJobORM, job_id)
            if row is None:
                raise ValueError(f"Job not found: {job_id}")
            row.status = status
            row.completed_at = completed_at
            row.result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
            row.error = error

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.session() as db:
            row = db.get(BatchJobORM, job_id)
        if row is None:
            return None
        return {
            "job_id": row.job_id,
            "status": row.status,
            "created_at": row.created_at,
            "completed_at": row.completed_at,
            "result": json.loads(row.result_json) if row.result_json else None,
            "error": row.error,
        }

    def record_sent_email(
        self,
        *,
        doc_id: str,
        recipients: List[str],
        subject: str,
        body: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        with self.session() as db:
            db.add(SentEmailORM(
                sent_at=_now_iso(),
                doc_id=doc_id,
                recipients_json=json.dumps(recipients, ensure_ascii=False),
                subject=subject,
                body=body,
                status=status,
                error=error,
            ))
