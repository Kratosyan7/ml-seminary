from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from datetime import datetime, UTC
import uuid
import logging
from .schemas import DocType, UploadResponse

ENV_PATH = Path(__file__).resolve().parents[2] / "notebooks" / ".env"
load_dotenv(ENV_PATH)  # Загружаем переменные окружения из файла .env

from .rag import RAGStore, load_text_from_file
from .quality import compute_demo_metrics, load_golden_dataset
from .persistence import PersistenceStore
from .emailer import SMTPEmailSender
from .security import SecurityHeadersMiddleware, SimpleRateLimitMiddleware, parse_allowed_origins, verify_api_key
from .monitoring import ANALYSIS_COUNT, EMAIL_SENT_COUNT, MetricsMiddleware, metrics_response
from .tasks import run_batch_analysis
from .schemas import (
    UploadResponse, AskRequest, AskResponse,
    ComplianceRequest, ComplianceVerdict,
    ContractChangeRequest, ContractChangeResponse,
    SendDraftsStubResponse, SendDraftsResponse,
    AuditRunListResponse, BatchJobResponse, DemoMetricsResponse,
)

logger = logging.getLogger("rag_agent")

app = FastAPI(title="Vacation Policy RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SimpleRateLimitMiddleware, max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "120")), window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
app.add_middleware(MetricsMiddleware)

store = RAGStore()
persistence = PersistenceStore()
email_sender = SMTPEmailSender()


def _sync_store_from_db(rebuild_index: bool = True) -> None:
    store.load_documents(persistence.list_documents(), rebuild_index=rebuild_index)


def _preload_demo_documents() -> None:
    if os.getenv("PRELOAD_DEMO_DOCS", "1") != "1":
        print("[preload] demo preload disabled")
        return

    repo_root = Path(__file__).resolve().parents[2]
    preload_files = [
        ("policy", "policy_main", repo_root / "notebooks" / "data" / "demo_policy.md"),
        ("contract", "contract_001", repo_root / "notebooks" / "contracts" / "contract_001.md"),
        ("contract", "contract_002", repo_root / "notebooks" / "contracts" / "contract_002.md"),
        ("contract", "contract_003", repo_root / "notebooks" / "contracts" / "contract_003.md"),
        ("contract", "contract_004", repo_root / "notebooks" / "contracts" / "contract_004.md"),
    ]

    for doc_type, doc_id, path in preload_files:
        if persistence.has_doc(doc_id):
            continue
        if not path.exists():
            print(f"[preload] skip missing file: {path}")
            continue

        text = load_text_from_file(path.name, path.read_bytes())
        persistence.upsert_document(
            doc_id=doc_id,
            doc_type=doc_type,
            filename=path.name,
            text=text,
        )
    _sync_store_from_db(rebuild_index=False)


_preload_demo_documents()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _needs_change_count(response: ContractChangeResponse) -> int:
    return sum(1 for item in response.results if item.needs_change)


def _run_batch_job(job_id: str, req: ContractChangeRequest) -> None:
    try:
        result = store.analyze_contract_changes(
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
        )
        completed_at = _now_iso()
        persistence.update_job(
            job_id=job_id,
            status="completed",
            completed_at=completed_at,
            result=result.model_dump(),
            error=None,
        )
        persistence.record_run(
            run_id=job_id,
            created_at=completed_at,
            event_type="batch_analysis",
            status="completed",
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
            payload=result.model_dump(),
            needs_change_count=_needs_change_count(result),
            error=None,
        )
        ANALYSIS_COUNT.labels(mode="batch_local", status="completed").inc()
    except Exception as exc:
        completed_at = _now_iso()
        persistence.update_job(
            job_id=job_id,
            status="failed",
            completed_at=completed_at,
            result=None,
            error=str(exc),
        )
        persistence.record_run(
            run_id=job_id,
            created_at=completed_at,
            event_type="batch_analysis",
            status="failed",
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
            payload=None,
            needs_change_count=None,
            error=str(exc),
        )
        ANALYSIS_COUNT.labels(mode="batch_local", status="failed").inc()

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/ready")
def ready():
    return {
        "ok": True,
        "db": persistence.ping(),
        "docs_loaded": len(store.docs),
        "queue_mode": "celery" if os.getenv("USE_CELERY_QUEUE", "0") == "1" else "background",
        "smtp_configured": email_sender.is_configured(),
    }


@app.get("/metrics")
def metrics():
    return metrics_response()

@app.get("/docs/list")
def list_docs():
    _sync_store_from_db()
    return {"docs": store.list_docs()}

@app.post("/docs/upload", response_model=UploadResponse)
async def upload_doc(
    request: Request,
    file: UploadFile = File(...),
    doc_type: DocType = Form(..., description="policy | contract | other"),
    doc_id: str | None = Form(None, description="опционально: свой идентификатор"),
    _: None = Depends(verify_api_key),
):
    try:
        content = await file.read()
        text = load_text_from_file(file.filename or "file", content)
        new_id = doc_id or f"{doc_type}_{uuid.uuid4().hex[:10]}"
        persistence.upsert_document(
            doc_id=new_id,
            doc_type=doc_type,
            filename=file.filename or "file",
            text=text,
        )
        _, chunks = store.upsert_text_document(
            doc_type=doc_type,
            filename=file.filename or "file",
            text=text,
            doc_id=new_id,
        )
        return UploadResponse(doc_id=new_id, doc_type=doc_type, chunks_indexed=chunks)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/docs/upload/batch")
async def upload_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    doc_type: str = Form(...),
    _: None = Depends(verify_api_key),
):
    results = []
    for f in files:
        content = await f.read()
        text = load_text_from_file(f.filename or "file", content)
        doc_id = f"{doc_type}_{uuid.uuid4().hex[:10]}"
        persistence.upsert_document(
            doc_type=doc_type,
            filename=f.filename or "file",
            text=text,
            doc_id=doc_id,
        )
        _, chunks = store.upsert_text_document(
            doc_type=doc_type,
            filename=f.filename or "file",
            text=text,
            doc_id=doc_id,
        )
        results.append({
            "doc_id": doc_id,
            "filename": f.filename,
            "chunks_indexed": chunks,
        })
    return {"uploaded": results}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer, citations = store.ask(req.question, doc_ids=req.doc_ids)
        used = req.doc_ids or [d["doc_id"] for d in store.list_docs()]
        return AskResponse(answer=answer, used_doc_ids=used, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/compliance/check", response_model=ComplianceVerdict)
def compliance_check(req: ComplianceRequest):
    try:
        return store.check_compliance(doc_ids=req.doc_ids, focus=req.focus)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/contracts/analyze-change", response_model=ContractChangeResponse)
def analyze_contract_change(req: ContractChangeRequest):
    if not persistence.has_doc(req.policy_doc_id, "policy"):
        raise HTTPException(status_code=400, detail=f"Policy документ не найден: {req.policy_doc_id}")

    missing_contracts = [doc_id for doc_id in req.contract_doc_ids if not persistence.has_doc(doc_id, "contract")]
    if missing_contracts:
        raise HTTPException(
            status_code=400,
            detail=f"Не найдены contract документы: {', '.join(missing_contracts)}"
        )

    try:
        _sync_store_from_db()
        result = store.analyze_contract_changes(
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
        )
        persistence.record_run(
            run_id=uuid.uuid4().hex,
            created_at=_now_iso(),
            event_type="sync_analysis",
            status="completed",
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
            payload=result.model_dump(),
            needs_change_count=_needs_change_count(result),
            error=None,
        )
        ANALYSIS_COUNT.labels(mode="sync", status="completed").inc()
        return result
    except Exception as e:
        persistence.record_run(
            run_id=uuid.uuid4().hex,
            created_at=_now_iso(),
            event_type="sync_analysis",
            status="failed",
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
            payload=None,
            needs_change_count=None,
            error=str(e),
        )
        ANALYSIS_COUNT.labels(mode="sync", status="failed").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/contracts/send-drafts-stub", response_model=SendDraftsStubResponse)
def send_drafts_stub(req: ContractChangeRequest):
    if not persistence.has_doc(req.policy_doc_id, "policy"):
        raise HTTPException(status_code=400, detail=f"Policy документ не найден: {req.policy_doc_id}")

    missing_contracts = [doc_id for doc_id in req.contract_doc_ids if not persistence.has_doc(doc_id, "contract")]
    if missing_contracts:
        raise HTTPException(
            status_code=400,
            detail=f"Не найдены contract документы: {', '.join(missing_contracts)}"
        )

    try:
        _sync_store_from_db()
        analysis = store.analyze_contract_changes(
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
        )
        drafts = store.collect_email_drafts(analysis)
        sent = store.send_email_stub(drafts)
        persistence.record_run(
            run_id=uuid.uuid4().hex,
            created_at=_now_iso(),
            event_type="send_drafts_stub",
            status="completed",
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
            payload={"drafts": [draft.model_dump() for draft in sent]},
            needs_change_count=len(sent),
            error=None,
        )
        return SendDraftsStubResponse(drafts=sent)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/contracts/send-drafts", response_model=SendDraftsResponse)
def send_drafts(req: ContractChangeRequest, _: None = Depends(verify_api_key)):
    if not persistence.has_doc(req.policy_doc_id, "policy"):
        raise HTTPException(status_code=400, detail=f"Policy документ не найден: {req.policy_doc_id}")

    missing_contracts = [doc_id for doc_id in req.contract_doc_ids if not persistence.has_doc(doc_id, "contract")]
    if missing_contracts:
        raise HTTPException(status_code=400, detail=f"Не найдены contract документы: {', '.join(missing_contracts)}")

    _sync_store_from_db()
    analysis = store.analyze_contract_changes(policy_doc_id=req.policy_doc_id, contract_doc_ids=req.contract_doc_ids)
    drafts = store.collect_email_drafts(analysis)

    sent: list = []
    for draft in drafts:
        try:
            email_sender.send(draft)
            persistence.record_sent_email(
                doc_id=draft.doc_id,
                recipients=draft.to,
                subject=draft.subject,
                body=draft.body,
                status="sent",
                error=None,
            )
            EMAIL_SENT_COUNT.labels(status="sent").inc()
            sent.append(draft)
        except Exception as exc:
            persistence.record_sent_email(
                doc_id=draft.doc_id,
                recipients=draft.to,
                subject=draft.subject,
                body=draft.body,
                status="failed",
                error=str(exc),
            )
            EMAIL_SENT_COUNT.labels(status="failed").inc()
            raise HTTPException(status_code=400, detail=f"Ошибка отправки email для {draft.doc_id}: {exc}")

    persistence.record_run(
        run_id=uuid.uuid4().hex,
        created_at=_now_iso(),
        event_type="send_drafts",
        status="completed",
        policy_doc_id=req.policy_doc_id,
        contract_doc_ids=req.contract_doc_ids,
        payload={"drafts": [draft.model_dump() for draft in sent]},
        needs_change_count=len(sent),
        error=None,
    )
    return SendDraftsResponse(sent=sent)


@app.post("/contracts/analyze-change/batch", response_model=BatchJobResponse)
def analyze_contract_change_batch(req: ContractChangeRequest, background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):
    if not persistence.has_doc(req.policy_doc_id, "policy"):
        raise HTTPException(status_code=400, detail=f"Policy документ не найден: {req.policy_doc_id}")

    missing_contracts = [doc_id for doc_id in req.contract_doc_ids if not persistence.has_doc(doc_id, "contract")]
    if missing_contracts:
        raise HTTPException(
            status_code=400,
            detail=f"Не найдены contract документы: {', '.join(missing_contracts)}"
        )

    job_id = uuid.uuid4().hex
    created_at = _now_iso()
    persistence.create_job(
        job_id=job_id,
        created_at=created_at,
        policy_doc_id=req.policy_doc_id,
        contract_doc_ids=req.contract_doc_ids,
    )
    if os.getenv("USE_CELERY_QUEUE", "0") == "1":
        run_batch_analysis.delay(job_id, req.policy_doc_id, req.contract_doc_ids)
    else:
        background_tasks.add_task(_run_batch_job, job_id, req)
    return BatchJobResponse(job_id=job_id, status="queued", created_at=created_at)


@app.get("/contracts/jobs/{job_id}", response_model=BatchJobResponse)
def get_batch_job(job_id: str):
    job = persistence.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job не найден: {job_id}")
    return BatchJobResponse(**job)


@app.get("/audit/runs", response_model=AuditRunListResponse)
def list_audit_runs(limit: int = 20):
    runs = persistence.list_runs(limit=limit)
    return AuditRunListResponse(runs=runs)


@app.get("/quality/demo-metrics", response_model=DemoMetricsResponse)
def quality_demo_metrics():
    repo_root = Path(__file__).resolve().parents[2]
    golden = load_golden_dataset(repo_root)
    _sync_store_from_db()
    result = store.analyze_contract_changes(
        policy_doc_id=golden["policy_doc_id"],
        contract_doc_ids=golden["contract_doc_ids"],
    )
    metrics = compute_demo_metrics(result, golden)
    return DemoMetricsResponse(**metrics)
