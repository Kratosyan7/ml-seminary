from pathlib import Path

from app.quality import compute_demo_metrics, load_golden_dataset
from app.rag import _extract_emails, _extract_number, _mask_email, _mask_name
from app.schemas import ContractChangeResponse, ContractChangeResult, PolicyRuleExtract


def test_extract_number_skips_clause_number():
    text = "4.3. Ежегодный оплачиваемый отпуск предоставляется продолжительностью не менее 14 календарных дней."
    assert _extract_number(text) == 14


def test_extract_multiple_emails():
    text = "Email: a@test.ru; резервный: b@test.ru"
    assert _extract_emails(text) == ["a@test.ru", "b@test.ru"]


def test_masking_helpers():
    assert _mask_email("ivanov.ii@company.ru").startswith("i***@")
    assert _mask_name("Иванов Иван Иванович").startswith("И***")


def test_demo_metrics():
    golden = load_golden_dataset(Path(__file__).resolve().parents[2])
    response = ContractChangeResponse(
        policy_rule=PolicyRuleExtract(rule_topic="x", new_value=10, unit="дней", source_quote="x"),
        results=[
            ContractChangeResult(doc_id="contract_001", emails=["ivanov.ii@company.ru"], old_value=7, new_value=10, unit="дней", needs_change=True, reason="x"),
            ContractChangeResult(doc_id="contract_002", emails=["maria.petrova@company.ru"], old_value=14, new_value=10, unit="дней", needs_change=False, reason="x"),
            ContractChangeResult(doc_id="contract_003", emails=["john.smith@company.ru"], old_value=8, new_value=10, unit="дней", needs_change=True, reason="x"),
            ContractChangeResult(doc_id="contract_004", emails=["anna.bolein@company.ru"], old_value=10, new_value=10, unit="дней", needs_change=False, reason="x"),
        ],
    )
    metrics = compute_demo_metrics(response, golden)
    assert metrics["classification"]["precision"] == 1.0
    assert metrics["classification"]["recall"] == 1.0
    assert metrics["extraction"]["old_value_accuracy"] == 1.0
    assert metrics["extraction"]["email_accuracy"] == 1.0
