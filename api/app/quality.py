import json
from pathlib import Path
from typing import Any, Dict, List

from .schemas import ContractChangeResponse


def load_golden_dataset(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "api" / "tests" / "golden_contract_change.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_demo_metrics(response: ContractChangeResponse, golden: Dict[str, Any]) -> Dict[str, Any]:
    expected = {item["doc_id"]: item for item in golden["contracts"]}
    predicted = {item.doc_id: item for item in response.results}

    tp = fp = fn = tn = 0
    old_value_matches = 0
    email_matches = 0
    total = 0

    per_doc: List[Dict[str, Any]] = []

    for doc_id, exp in expected.items():
        pred = predicted.get(doc_id)
        if pred is None:
            continue

        total += 1
        exp_change = bool(exp["needs_change"])
        pred_change = bool(pred.needs_change)

        if exp_change and pred_change:
            tp += 1
        elif not exp_change and pred_change:
            fp += 1
        elif exp_change and not pred_change:
            fn += 1
        else:
            tn += 1

        if pred.old_value == exp["old_value"]:
            old_value_matches += 1
        if sorted(pred.emails) == sorted(exp["emails"]):
            email_matches += 1

        per_doc.append({
            "doc_id": doc_id,
            "expected_change": exp_change,
            "predicted_change": pred_change,
            "expected_old_value": exp["old_value"],
            "predicted_old_value": pred.old_value,
            "expected_emails": exp["emails"],
            "predicted_emails": pred.emails,
        })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "classification": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "extraction": {
            "old_value_accuracy": old_value_matches / total if total else 0.0,
            "email_accuracy": email_matches / total if total else 0.0,
        },
        "total_contracts": total,
        "per_doc": per_doc,
    }
