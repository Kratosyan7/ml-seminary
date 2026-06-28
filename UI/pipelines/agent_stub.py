"""
Open WebUI Pipeline: Contract Change Agent
Ходит в backend FastAPI и показывает отчет по договорам.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional, Dict, Any

class Pipeline:
    name = "Contract Change Agent"
    id = "agent_stub"

    version = "0.2.1"
    description = "Анализирует договоры на соответствие policy через backend FastAPI."

    def __init__(self):
        self.api_base_url = os.getenv("AGENT_API_BASE_URL", "http://host.docker.internal:8000").rstrip("/")
        self.request_timeout = int(os.getenv("AGENT_API_TIMEOUT_SECONDS", "120"))
        self.model_name = os.getenv("AGENT_MODEL_NAME", "qwen2.5:7b")

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

    def _is_tag_generation_task(self, text: str) -> bool:
        normalized = (text or "").lower()
        return (
            "generate 1-3 broad tags" in normalized
            and '"tags"' in normalized
            and "<chat_history>" in normalized
        )

    def _is_title_generation_task(self, text: str) -> bool:
        normalized = (text or "").lower()
        return (
            "generate a concise, 3-5 word title" in normalized
            and '"title"' in normalized
            and "<chat_history>" in normalized
        )

    def _is_followups_generation_task(self, text: str) -> bool:
        normalized = (text or "").lower()
        return (
            "suggest 3-5 relevant follow-up questions" in normalized
            and '"follow_ups"' in normalized
            and "<chat_history>" in normalized
        )

    def _build_tag_response(self, text: str) -> str:
        normalized = (text or "").lower()
        if any(token in normalized for token in ["договор", "policy", "отпуск", "contract"]):
            return json.dumps({"tags": ["Документы", "Отпуск", "Policy"]}, ensure_ascii=False)
        return json.dumps({"tags": ["General"]}, ensure_ascii=False)

    def _build_title_response(self, text: str) -> str:
        normalized = (text or "").lower()
        if "договор" in normalized:
            return json.dumps({"title": "📄 Проверка договоров"}, ensure_ascii=False)
        if "отпуск" in normalized:
            return json.dumps({"title": "🏖️ Вопросы по отпуску"}, ensure_ascii=False)
        return json.dumps({"title": "💬 Диалог"}, ensure_ascii=False)

    def _build_followups_response(self, text: str) -> str:
        normalized = (text or "").lower()
        if "договор" in normalized or "policy" in normalized:
            payload = {
                "follow_ups": [
                    "Какие договоры нужно изменить?",
                    "Какое новое правило указано в policy?",
                    "Сформируй черновики уведомлений."
                ]
            }
            return json.dumps(payload, ensure_ascii=False)
        payload = {
            "follow_ups": [
                "Какая минимальная длина одной части отпуска?",
                "За сколько дней до начала отпуска нужно подать заявление?",
                "Проверь договоры на соответствие новой политике отпусков."
            ]
        }
        return json.dumps(payload, ensure_ascii=False)

    def _is_analysis_request(self, text: str) -> bool:
        normalized = (text or "").lower()
        strong_triggers = [
            "проверь договор",
            "проверь договоры",
            "проанализируй договор",
            "проанализируй договоры",
            "сравни договор",
            "сравни договоры",
            "соответствие политике",
            "соответствие новой политике",
            "проверь на соответствие",
            "нужно ли менять договор",
            "какие договоры нужно изменить",
            "уведомлен",
        ]
        return any(token in normalized for token in strong_triggers)

    def _is_general_question(self, text: str) -> bool:
        normalized = (text or "").lower().strip()
        question_starts = [
            "какая",
            "какой",
            "какие",
            "сколько",
            "когда",
            "как",
            "где",
            "за сколько",
            "нужно ли",
            "что",
        ]
        soft_question_tokens = [
            "отпуск",
            "отпуска",
            "заявление",
            "заявления",
            "политика",
            "policy",
            "часть отпуска",
        ]
        return (
            any(normalized.startswith(token) for token in question_starts)
            or normalized.endswith("?")
            or any(token in normalized for token in soft_question_tokens)
        )

    def _is_greeting(self, text: str) -> bool:
        normalized = (text or "").lower().strip()
        return normalized in {"привет", "здравствуй", "здравствуйте", "hi", "hello", "hey"}

    def _is_meta_question(self, text: str) -> bool:
        normalized = (text or "").lower().strip()
        markers = [
            "какая модель",
            "какой моделью",
            "что за модель",
            "какая у тебя модель",
            "какой у тебя backend",
            "как ты работаешь",
            "что ты умеешь",
        ]
        return any(token in normalized for token in markers)

    def _build_meta_response(self) -> str:
        return (
            "Я pipeline `Contract Change Agent`.\n"
            f"- Backend: FastAPI ({self.api_base_url})\n"
            f"- LLM: {self.model_name} через Ollama\n"
            "- Режим 1: вопросы по кадровым документам через RAG\n"
            "- Режим 2: анализ договоров на соответствие policy"
        )

    def _extract_doc_ids(self, text: str) -> Dict[str, List[str]]:
        content = text or ""
        policy_ids = re.findall(r"\bpolicy_[A-Za-z0-9_]+\b", content)
        contract_ids = re.findall(r"\bcontract_[A-Za-z0-9_]+\b", content)
        return {
            "policy_ids": list(dict.fromkeys(policy_ids)),
            "contract_ids": list(dict.fromkeys(contract_ids)),
        }

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

        if self._is_tag_generation_task(user_msg):
            return self._build_tag_response(user_msg)
        if self._is_title_generation_task(user_msg):
            return self._build_title_response(user_msg)
        if self._is_followups_generation_task(user_msg):
            return self._build_followups_response(user_msg)

        if self._is_greeting(user_msg):
            return (
                "Привет. Я умею 2 режима: отвечать на вопросы по кадровым документам "
                "и проверять договоры на соответствие policy.\n\n"
                "Примеры:\n"
                "- Какая минимальная длина одной части отпуска?\n"
                "- За сколько дней до начала отпуска нужно подать заявление?\n"
                "- Проверь договоры на соответствие новой политике отпусков"
            )

        if self._is_meta_question(user_msg):
            return self._build_meta_response()

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
        doc_refs = self._extract_doc_ids(user_msg)

        if not policy_docs or not contract_docs:
            return (
                "Для анализа сначала загрузи документы в backend: минимум один `policy` и один `contract`. "
                "Сделай это через Swagger `/docs` или endpoint `/docs/upload`."
            )

        selected_policy_docs = policy_docs
        if doc_refs["policy_ids"]:
            selected_policy_docs = [
                item for item in policy_docs if item.get("doc_id") in doc_refs["policy_ids"]
            ]
            if not selected_policy_docs:
                return (
                    "Не нашел указанный policy в backend. "
                    f"Запрошено: {', '.join(doc_refs['policy_ids'])}."
                )

        selected_contract_docs = contract_docs
        if doc_refs["contract_ids"]:
            selected_contract_docs = [
                item for item in contract_docs if item.get("doc_id") in doc_refs["contract_ids"]
            ]
            if not selected_contract_docs:
                return (
                    "Не нашел указанные contract в backend. "
                    f"Запрошено: {', '.join(doc_refs['contract_ids'])}."
                )
        elif (
            doc_refs["policy_ids"]
            and selected_policy_docs[0].get("doc_id") != "policy_main"
        ):
            non_demo_contracts = [
                item for item in contract_docs
                if item.get("doc_id") not in {
                    "contract_001",
                    "contract_002",
                    "contract_003",
                    "contract_004",
                }
            ]
            if non_demo_contracts:
                selected_contract_docs = non_demo_contracts

        payload = {
            "policy_doc_id": selected_policy_docs[0]["doc_id"],
            "contract_doc_ids": [item["doc_id"] for item in selected_contract_docs],
        }

        if self._is_general_question(user_msg) and not self._is_analysis_request(user_msg):
            try:
                result = self._request("POST", "/ask", {"question": user_msg})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                return f"Backend вернул ошибку {exc.code}: {detail}"
            except urllib.error.URLError as exc:
                return f"Backend недоступен: {exc}"
            except Exception as exc:
                return f"Ошибка ответа на вопрос: {exc}"

            answer = result.get("answer") or "Ответ не получен."
            citations = result.get("citations") or []
            if citations:
                answer += "\n\nИсточники:\n- " + "\n- ".join(citations)
            return answer

        if not self._is_analysis_request(user_msg):
            return (
                "Не понял запрос. Спроси полным вопросом про документы "
                "или попроси проверить договоры на соответствие policy."
            )

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
