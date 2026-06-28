from pydantic import BaseModel, Field
from typing import List, Optional, Literal

DocType = Literal["policy", "contract", "other"]

class UploadResponse(BaseModel):
    doc_id: str
    doc_type: DocType
    chunks_indexed: int

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Если не указано — поиск идёт по всем загруженным документам."
    )

class AskResponse(BaseModel):
    answer: str
    used_doc_ids: List[str]
    citations: List[str] = Field(default_factory=list)

class ComplianceRequest(BaseModel):
    doc_ids: Optional[List[str]] = None
    focus: str = Field(
        default="отпуска",
        description="Фокус проверки, например: 'минимальная длительность отпуска', 'перенос', 'оплата', ...",
    )

class ComplianceVerdict(BaseModel):
    compliant: bool = Field(description="Соответствуют ли документы политике")
    summary: str = Field(description="Короткий вывод в 1-3 предложениях")
    violations: List[str] = Field(default_factory=list, description="Список нарушений/несоответствий")
    evidence: List[str] = Field(default_factory=list, description="Цитаты/фрагменты из документов и политики")
    recommendations: List[str] = Field(default_factory=list, description="Что исправить/уточнить")


class PolicyRuleExtract(BaseModel):
    rule_topic: str = Field(description="Тема правила, например: минимальная длительность части отпуска")
    new_value: Optional[int] = Field(default=None, description="Новое значение правила числом")
    unit: Optional[str] = Field(default=None, description="Единица измерения, например: календарных дней")
    source_quote: Optional[str] = Field(default=None, description="Короткая цитата из policy")


class ContractExtract(BaseModel):
    employee_name: Optional[str] = Field(default=None, description="ФИО сотрудника")
    email: Optional[str] = Field(default=None, description="Email для уведомления")
    emails: List[str] = Field(default_factory=list, description="Все email адреса, явно найденные в договоре")
    current_value: Optional[int] = Field(default=None, description="Текущее значение условия в договоре")
    unit: Optional[str] = Field(default=None, description="Единица измерения")
    source_quote: Optional[str] = Field(default=None, description="Короткая цитата из договора")
    evidence: Optional[str] = Field(default=None, description="Фрагмент текста, подтверждающий текущее условие")


class ContractChangeResult(BaseModel):
    doc_id: str
    employee_name: Optional[str] = None
    email: Optional[str] = None
    emails: List[str] = Field(default_factory=list)
    old_value: Optional[int] = None
    new_value: Optional[int] = None
    unit: Optional[str] = None
    needs_change: bool
    reason: str
    source_quote: Optional[str] = None
    evidence: Optional[str] = None
    draft_subject: Optional[str] = None
    draft_body: Optional[str] = None


class ContractChangeRequest(BaseModel):
    policy_doc_id: str = Field(description="ID policy-документа с новым правилом")
    contract_doc_ids: List[str] = Field(description="Список ID договоров для проверки")


class ContractChangeResponse(BaseModel):
    policy_rule: PolicyRuleExtract
    results: List[ContractChangeResult]


class EmailDraft(BaseModel):
    doc_id: str
    to: List[str] = Field(default_factory=list)
    subject: str
    body: str


class EmailDraftContent(BaseModel):
    subject: str = Field(description="Короткая тема письма без markdown и без префикса Subject")
    body: str = Field(description="Тело письма на русском, нейтральное, деловое, без выдуманных вложений, сроков и placeholder'ов")


class SendDraftsStubResponse(BaseModel):
    drafts: List[EmailDraft]


class SendDraftsResponse(BaseModel):
    sent: List[EmailDraft]


class AuditRun(BaseModel):
    id: str
    created_at: str
    event_type: str
    status: str
    policy_doc_id: Optional[str] = None
    contract_doc_ids: List[str] = Field(default_factory=list)
    needs_change_count: Optional[int] = None
    payload: Optional[dict] = None
    error: Optional[str] = None


class AuditRunListResponse(BaseModel):
    runs: List[AuditRun]


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[ContractChangeResponse] = None
    error: Optional[str] = None


class DemoMetricsResponse(BaseModel):
    classification: dict
    extraction: dict
    total_contracts: int
    per_doc: List[dict]
