"""
Open WebUI Pipeline: Contract Change Agent
Ходит в backend FastAPI и показывает отчет по договорам.
"""

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional, Dict, Any

class Pipeline:
    name = "Contract Change Agent"
    id = "agent_stub"

    version = "0.2.0"
    description = "Анализирует договоры на соответствие policy через backend FastAPI."

    def __init__(self):
        self.api_base_url = os.getenv("AGENT_API_BASE_URL", "http://host.docker.internal:8000").rstrip("/")
        self.request_timeout = int(os.getenv("AGENT_API_TIMEOUT_SECONDS", "120"))

    async def on_startup(self):
        print(f"[agent_stub] startup, api={self.api_base_url}")

    async def on_shutdown(self):
        print("[agent_stub] shutdown")

    def _extract_user_message(self, body: Dict[str, Any]) -> str:
        try:
            msgs = body.get("messages", [])
            if msgs:
                return msgs[-1].get("content", "")
        except Exception:
            return ""
        return ""

    def _is_analysis_request(self, text: str) -> bool:
        normalized = (text or "").lower()
        triggers = [
            "договор",
            "договоры",
            "отпуск",
            "политик",
            "уведомлен",
            "проверь",
            "проанализируй",
            "сравни",
        ]
        return any(token in normalized for token in triggers)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url=f"{self.api_base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _format_analysis(self, payload: Dict[str, Any]) -> str:
        policy_rule = payload.get("policy_rule", {})
        results = payload.get("results", [])
        needs_change_count = sum(1 for item in results if item.get("needs_change"))

        lines = [
            f"Проверено договоров: {len(results)}.",
            f"Требуют изменений: {needs_change_count}.",
            "",
            "Правило policy:",
            f"- Тема: {policy_rule.get('rule_topic') or 'не указано'}",
            f"- Новое значение: {policy_rule.get('new_value')} {policy_rule.get('unit') or ''}".strip(),
            f"- Цитата: {policy_rule.get('source_quote') or 'нет'}",
            "",
            "Результаты:",
        ]

        for item in results:
            status = "нужно изменить" if item.get("needs_change") else "ok"
            emails = item.get("emails") or []
            lines.extend([
                f"- {item.get('doc_id')}: {status}",
                f"  Сотрудник: {item.get('employee_name') or 'не найден'}",
                f"  Emails: {', '.join(emails) or item.get('email') or 'не найдены'}",
                f"  Старое значение: {item.get('old_value')}",
                f"  Новое значение: {item.get('new_value')}",
                f"  Причина: {item.get('reason')}",
            ])

        drafts = [item for item in results if item.get("needs_change")]
        if drafts:
            lines.extend(["", "Черновики уведомлений:"])
            for item in drafts:
                lines.extend([
                    f"- {item.get('doc_id')}",
                    f"  Тема: {item.get('draft_subject') or 'без темы'}",
                    item.get("draft_body") or "Черновик не сформирован.",
                    "",
                ])

        return "\n".join(lines).strip()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__=None,
        **kwargs, 
    ):
        user_msg = self._extract_user_message(body)
        print(f"[agent_stub] user said: {user_msg!r}")

        if not self._is_analysis_request(user_msg):
            return (
                "Этот pipeline предназначен для анализа договоров по policy. "
                "Напиши запрос вроде: 'Проверь договоры на соответствие новой политике отпусков'."
            )

        try:
            docs_payload = self._request("GET", "/docs/list")
        except urllib.error.URLError as exc:
            return (
                "Не удалось подключиться к backend FastAPI. "
                f"Проверь AGENT_API_BASE_URL={self.api_base_url}. Ошибка: {exc}"
            )
        except Exception as exc:
            return f"Ошибка при обращении к backend: {exc}"

        docs = docs_payload.get("docs", [])
        policy_docs = [item for item in docs if item.get("doc_type") == "policy"]
        contract_docs = [item for item in docs if item.get("doc_type") == "contract"]

        if not policy_docs or not contract_docs:
            return (
                "Для анализа сначала загрузи документы в backend: минимум один `policy` и один `contract`. "
                "Сделай это через Swagger `/docs` или endpoint `/docs/upload`."
            )

        payload = {
            "policy_doc_id": policy_docs[0]["doc_id"],
            "contract_doc_ids": [item["doc_id"] for item in contract_docs],
        }

        try:
            result = self._request("POST", "/contracts/analyze-change", payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            return f"Backend вернул ошибку {exc.code}: {detail}"
        except urllib.error.URLError as exc:
            return f"Backend недоступен: {exc}"
        except Exception as exc:
            return f"Ошибка анализа: {exc}"

        return self._format_analysis(result)
