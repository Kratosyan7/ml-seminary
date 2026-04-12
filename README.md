# RAG Contract Change Agent

Проект показывает, как из базового `RAG` сделать прикладного агента для кадровых документов.

Система:

- читает policy компании
- читает трудовые договоры
- находит условие про отпуск
- сравнивает старое условие с новым правилом
- формирует черновики уведомлений для договоров, которые нужно обновить


## Что делает проект

Идея простая:

1. Есть документ с новым правилом компании.
2. Есть несколько трудовых договоров.
3. Агент находит нужные фрагменты в документах через `RAG`.
4. Извлекает структуру:
   - имя сотрудника
   - email
   - текущее значение условия
   - новое значение из policy
5. Сравнение делает кодом, а не LLM.
6. Если договор не соответствует policy, агент готовит `draft` письма.

Важно:

- письма не отправляются автоматически
- система делает только черновики
- решение работает локально через `Ollama`


## Архитектура

В проекте есть 3 основные части.

### 1. Backend на FastAPI

Backend:

- хранит документы
- строит `FAISS` индекс
- делает retrieval
- вызывает локальную модель через `Ollama`
- сравнивает условия
- возвращает JSON-результат

Главные файлы:

- [main.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/main.py)
- [rag.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/rag.py)
- [schemas.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api/app/schemas.py)

### 2. Локальная модель через Ollama

`Ollama` используется для:

- извлечения правила из policy
- извлечения имени, email и текущего условия из договора
- генерации черновика письма

### 3. Интерфейс через Open WebUI

`Open WebUI` нужен как удобный чат-интерфейс для демонстрации.

Файл:

- [agent_stub.py](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/UI/pipelines/agent_stub.py)

Он получает запрос из UI, обращается в backend и показывает итоговый ответ пользователю.


## Demo-документы

Для демонстрации используются файлы:

- [demo_policy.md](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/data/demo_policy.md)
- [contract_001.md](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/contracts/contract_001.md)
- [contract_002.md](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/contracts/contract_002.md)
- [contract_003.md](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/contracts/contract_003.md)
- [contract_004.md](/Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/notebooks/contracts/contract_004.md)

При старте backend эти demo-документы можно автоматически подгружать через `PRELOAD_DEMO_DOCS=1`.


## Как запустить

### 1. Подготовить Python-окружение

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary
python3 -m venv .venv
source .venv/bin/activate
cd api
python3 -m pip install -r requirements.txt
```

### 2. Запустить Ollama

В отдельном окне терминала:

```bash
ollama serve
```

Если модель еще не скачана:

```bash
ollama pull qwen2.5:1.5b
```

### 3. Запустить backend

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/api
source ../.venv/bin/activate
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

### 4. Запустить Open WebUI

```bash
cd /Users/ashotmirzoyan/Documents/ML-Seminary/ml-seminary/UI
docker compose up
```

После запуска:

```text
http://localhost:3000
```


## Что написать в чате

В `Open WebUI` можно отправить запрос:

```text
Проверь договоры на соответствие новой политике отпусков
```

Ожидаемое поведение:

- система покажет summary по 4 договорам
- `contract_001` и `contract_003` будут помечены как требующие изменений
- `contract_002` и `contract_004` будут отмечены как соответствующие policy
- для проблемных договоров появятся черновики уведомлений


## Главный endpoint

Основной endpoint backend:

### `POST /contracts/analyze-change`

Пример запроса:

```bash
curl -s -X POST "http://localhost:8000/contracts/analyze-change" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_doc_id": "policy_main",
    "contract_doc_ids": ["contract_001", "contract_002", "contract_003", "contract_004"]
  }'
```

Дополнительно есть endpoint-заглушка для черновиков:

### `POST /contracts/send-drafts-stub`

Он ничего реально не отправляет, а только возвращает подготовленные draft-письма и печатает их в лог.


## Ожидаемый результат demo

Политика:

- минимум 10 дней

Результат сравнения:

- `contract_001` → 7 дней → нужно изменение
- `contract_002` → 14 дней → соответствует
- `contract_003` → 8 дней → нужно изменение
- `contract_004` → 10 дней → соответствует


## Что важно сказать на защите

1. Это не просто чат по документам, а прикладной `RAG`-агент.
2. Retrieval используется для поиска релевантных фрагментов policy и договоров.
3. `LLM` используется для извлечения структуры и генерации черновиков.
4. Критическая бизнес-логика сравнения сделана кодом, а не LLM.
5. Письма не отправляются автоматически, только формируются как `draft`.
6. Вся система работает локально через `Ollama`, без OpenAI API.


## Ограничения текущей версии

- документы и индекс хранятся в памяти
- нет постоянного хранилища результатов
- нет реальной рассылки
- нет production-механизмов вроде очередей, аудита и контроля доступа
- тексты писем сделаны как demo-черновики


## Коротко

Это локальный проект, который читает policy и трудовые договоры, находит несоответствия и готовит черновики уведомлений.
