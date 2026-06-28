import os
from datetime import datetime, UTC

from celery import Celery

from .persistence import PersistenceStore
from .rag import RAGStore


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


broker_url = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://redis:6379/0"))
backend_url = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://redis:6379/1"))
celery_app = Celery("contract_change_agent", broker=broker_url, backend=backend_url)


@celery_app.task(name="contract_change_agent.run_batch_analysis")
def run_batch_analysis(job_id: str, policy_doc_id: str, contract_doc_ids: list[str]) -> dict:
    persistence = PersistenceStore()
    store = RAGStore()
    store.load_documents(persistence.list_documents())

    try:
        persistence.update_job(job_id=job_id, status="running")
        result = store.analyze_contract_changes(policy_doc_id=policy_doc_id, contract_doc_ids=contract_doc_ids)
        payload = result.model_dump()
        persistence.update_job(
            job_id=job_id,
            status="completed",
            completed_at=_now_iso(),
            result=payload,
            error=None,
        )
        persistence.record_run(
            run_id=job_id,
            created_at=_now_iso(),
            event_type="batch_analysis",
            status="completed",
            policy_doc_id=policy_doc_id,
            contract_doc_ids=contract_doc_ids,
            payload=payload,
            needs_change_count=sum(1 for item in result.results if item.needs_change),
            error=None,
        )
        return payload
    except Exception as exc:
        persistence.update_job(
            job_id=job_id,
            status="failed",
            completed_at=_now_iso(),
            result=None,
            error=str(exc),
        )
        persistence.record_run(
            run_id=job_id,
            created_at=_now_iso(),
            event_type="batch_analysis",
            status="failed",
            policy_doc_id=policy_doc_id,
            contract_doc_ids=contract_doc_ids,
            payload=None,
            needs_change_count=None,
            error=str(exc),
        )
        raise
