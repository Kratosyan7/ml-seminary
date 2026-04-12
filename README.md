# RAG Contract Change Agent

Локальный demo-проект для проверки трудовых договоров на соответствие новой policy компании по отпускам.

Система:
- загружает `policy` и `contract` документы
- строит `FAISS` индекс
- через `Ollama` извлекает правило из policy и значения из договоров
- кодом сравнивает условия
- генерирует `draft` уведомлений для договоров, где нужно изменение
- показывает результат через `FastAPI` и `Open WebUI`

## Архитектура

Поток такой:

1. `FastAPI` backend хранит документы и индекс.
2. `RAGStore` ищет релевантные чанки.
3. `Ollama` извлекает структуру из контекста.
4. Python-код сравнивает старое и новое значение.
5. `Open WebUI pipeline` вызывает backend и показывает отчет в чате.

Основные части проекта:
- [api/app/main.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/main.py) — API и автозагрузка demo-документов
- [api/app/rag.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/rag.py) — RAG, extraction, compare, draft generation
- [api/app/schemas.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/schemas.py) — Pydantic-схемы
- [UI/pipelines/agent_stub.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/UI/pipelines/agent_stub.py) — pipeline-клиент backend
- [UI/docker-compose.yml](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/UI/docker-compose.yml) — запуск `Open WebUI`

## Требования

- Python 3.11+
- `Ollama`
- Docker Desktop

## Настройка

### 1. Python-зависимости

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary
python3 -m venv .venv
source .venv/bin/activate
cd api
python3 -m pip install -r requirements.txt
```

### 2. Ollama

Запустить сервер:

```bash
ollama serve
```

Скачать модель:

```bash
ollama pull qwen2.5:1.5b
```

### 3. Конфиг

Используется файл [notebooks/.env](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/.env).

Ключевые переменные:

```env
USE_OPENAI=0
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen2.5:1.5b
USE_OLLAMA_EMBEDDINGS=0
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
PRELOAD_DEMO_DOCS=1
```

Если `PRELOAD_DEMO_DOCS=1`, backend сам подгрузит demo policy и 4 договора при старте.

## Запуск

### Backend

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs/list
```

Swagger:

```text
http://localhost:8000/docs
```

### Open WebUI

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/UI
docker compose up
```

UI:

```text
http://localhost:3000
```

## Главный endpoint

### `POST /contracts/analyze-change`

Request:

```json
{
  "policy_doc_id": "policy_main",
  "contract_doc_ids": ["contract_001", "contract_002", "contract_003", "contract_004"]
}
```

Response содержит:
- извлеченное правило из policy
- список результатов по договорам
- `needs_change`
- причину
- draft письма для проблемных договоров

Пример вызова:

```bash
curl -s -X POST "http://localhost:8000/contracts/analyze-change" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_doc_id": "policy_main",
    "contract_doc_ids": ["contract_001", "contract_002", "contract_003", "contract_004"]
  }'
```

## Ожидаемый demo-результат

- `contract_001` → `7 < 10` → нужно изменение
- `contract_002` → `14 >= 10` → ok
- `contract_003` → `8 < 10` → нужно изменение
- `contract_004` → `10 >= 10` → ok

Для `contract_001` и `contract_003` должны появиться черновики уведомлений.

## Что говорить на защите

Короткий сценарий:

1. Это не просто чат с документами, а прикладной RAG-агент.
2. В backend загружаются policy и договоры, дальше строится `FAISS` индекс.
3. `Ollama` извлекает структуру из найденного контекста: правило policy, имя, email и текущее значение из договора.
4. Сравнение делает не LLM, а Python-код. Это делает решение детерминированным.
5. Если договор не соответствует policy, система не отправляет письма автоматически, а готовит только `draft`.
6. В UI пользователь получает сводку по договорам и черновики уведомлений.

## Ограничения текущей версии

- индекс и документы хранятся в памяти
- пока нет постоянного хранилища
- авторассылки нет
- extraction завязан на demo-формулировки и может требовать доработки для более шумных документов
- `MCP` в этой версии не используется
