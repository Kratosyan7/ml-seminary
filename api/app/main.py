from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from .schemas import DocType, UploadResponse

ENV_PATH = Path(__file__).resolve().parents[2] / "notebooks" / ".env"
load_dotenv(ENV_PATH)  # Загружаем переменные окружения из файла .env

from .rag import RAGStore
from .schemas import (
    UploadResponse, AskRequest, AskResponse,
    ComplianceRequest, ComplianceVerdict,
    ContractChangeRequest, ContractChangeResponse,
)

app = FastAPI(title="Vacation Policy RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ограничить при необходимости
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RAGStore()


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
        if store.has_doc(doc_id):
            continue
        if not path.exists():
            print(f"[preload] skip missing file: {path}")
            continue

        content = path.read_bytes()
        store.upsert_document(
            doc_type=doc_type,
            filename=path.name,
            content=content,
            doc_id=doc_id,
        )


_preload_demo_documents()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/docs/list")
def list_docs():
    return {"docs": store.list_docs()}

@app.post("/docs/upload", response_model=UploadResponse)
async def upload_doc(
    file: UploadFile = File(...),
    doc_type: DocType = Form(..., description="policy | contract | other"),
    doc_id: str | None = Form(None, description="опционально: свой идентификатор"),
):
    try:
        content = await file.read()
        new_id, chunks = store.upsert_document(
            doc_type=doc_type,
            filename=file.filename or "file",
            content=content,
            doc_id=doc_id
        )
        return UploadResponse(doc_id=new_id, doc_type=doc_type, chunks_indexed=chunks)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/docs/upload/batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    doc_type: str = Form(...),
):
    results = []
    for f in files:
        content = await f.read()
        doc_id, chunks = store.upsert_document(
            doc_type=doc_type,
            filename=f.filename or "file",
            content=content,
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
    if not store.has_doc(req.policy_doc_id, "policy"):
        raise HTTPException(status_code=400, detail=f"Policy документ не найден: {req.policy_doc_id}")

    missing_contracts = [doc_id for doc_id in req.contract_doc_ids if not store.has_doc(doc_id, "contract")]
    if missing_contracts:
        raise HTTPException(
            status_code=400,
            detail=f"Не найдены contract документы: {', '.join(missing_contracts)}"
        )

    try:
        return store.analyze_contract_changes(
            policy_doc_id=req.policy_doc_id,
            contract_doc_ids=req.contract_doc_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
