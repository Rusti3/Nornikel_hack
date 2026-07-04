# Nornikel Knowledge Graph / Agentic RAG

Локальный Docker-стек для демо “Научный клубок”: Neo4j knowledge graph, Postgres/ParadeDB vector+BM25 search, backend Agentic RAG worker и frontend-чат.

## Что нужно установить

- Docker Desktop.
- Git.
- Git LFS: <https://git-lfs.com/>.
- PowerShell 7 или стандартный Windows PowerShell.

Проверьте, что Docker запущен:

```powershell
docker info
git lfs version
```

## Быстрый запуск

```powershell
git clone https://github.com/Rusti3/Nornikel_hack.git
cd Nornikel_hack
git lfs pull

Copy-Item .env.example .env
notepad .env
```

В `.env` заполните минимум:

```dotenv
YANDEX_API_KEY=<ваш ключ Yandex AI Studio>
YANDEX_FOLDER_ID=<ваш Folder ID>
LLM_MODEL=gpt://<ваш Folder ID>/aliceai-llm/latest
EMBED_DOC_MODEL=emb://<ваш Folder ID>/text-embeddings-v2-doc/latest
EMBED_QUERY_MODEL=emb://<ваш Folder ID>/text-embeddings-v2-query/latest
NEO4J_PASSWORD=<любой надежный пароль>
POSTGRES_PASSWORD=<любой надежный пароль>
```

Через Git LFS загружаются готовые seed-дампы Postgres и Neo4j, а также две
демонстрационные презентации для Q01 и Q04. Полный 4,86-ГБ корпус не нужен для
демо и в репозиторий не входит.

Загрузка готовых Docker-образов и запуск:

```powershell
docker compose pull
.\start.ps1
```

При первом запуске пустые Docker volumes автоматически наполняются готовыми
данными: Postgres/ParadeDB восстанавливает chunks, BM25 и embeddings, Neo4j —
knowledge graph. Повторные запуски существующие volumes не перезаписывают.

Открыть:

- Agentic RAG chat: <http://localhost:8080>
- Backend health: <http://localhost:8000/health>
- Backend OpenAPI: <http://localhost:8000/docs>
- Neo4j Browser: <http://localhost:7474>
  - login: `neo4j`
  - password: значение `NEO4J_PASSWORD` из `.env`

Остановка:

```powershell
.\stop.ps1
```

Главная страница специально упрощена до одного чата. Единственная настройка — переключатель `Web Search`, по умолчанию выключенный. Старый Graph Builder сохранён на скрытом маршруте <http://localhost:8080/builder>.

## Что показывать на демо

В UI откройте чат Agentic RAG. Web Search оставьте выключенным, чтобы ответы строились только по локальным рабочим документам.

### Q01 — числовой факт из PPTX

```text
Какие извлечения Co, Ni, Cu и Mn заявлены для процесса Cuprion при переработке кобальт-марганцевых корок и на каких операциях основана схема?
```

Ожидаемо: `full_answer`, таблица с Co 91%, Ni 85%, Cu 61%, Mn 60%, ссылка на презентацию Цымбулова / слайд 19.

### Q04 — аналитический ответ по МПГ

```text
Почему штейновая плавка с последующим конвертированием названа предпочтительной головной операцией для МПГ-сырья и какие направления исследований остаются актуальными?
```

Ожидаемо: связный ответ с citations по штейновой плавке, конвертированию, извлечению МПГ и направлениям исследований.

## Локальная сборка без GHCR

Обычный запуск использует готовые образы:

- `ghcr.io/rusti3/nornikel_hack/backend:latest`
- `ghcr.io/rusti3/nornikel_hack/frontend:latest`

Если GHCR недоступен или нужно собрать код самостоятельно:

```powershell
.\start.ps1 -Build
```

Чистая сборка backend устанавливает тяжёлые Python/ML-зависимости и может занять продолжительное время.

## Полная индексация корпуса

Полный исходный корпус намеренно не входит в репозиторий. Если нужно
пересобрать индекс/граф на собственном наборе документов, укажите его папку в
`.env`:

```dotenv
CORPUS_PATH=C:/path/to/your/documents
```

Затем запустите:

```powershell
.\run-mekg-full.ps1 -Action Start
.\run-mekg-full.ps1 -Action Status -RunId <run-id>
```

Это долгий процесс и зависит от квот Yandex AI Studio. Для быстрого демо сначала проверьте готовность сервисов и чатовые сценарии выше.

## Полезные команды

```powershell
docker compose ps
docker compose logs -f backend
docker compose logs -f agent-worker
docker compose config --quiet
```

Обновление готовых образов:

```powershell
docker compose pull
docker compose up -d
```

Backend-тесты:

```powershell
docker compose run --rm --no-deps backend pytest -q
```

Проверка Yandex embeddings/LLM выполняется из приложения при запуске Agentic RAG. Векторы должны иметь размерность `768`.

## Безопасность

- Реальные ключи хранятся только в локальном `.env`.
- `.env`, artifacts, logs и Docker volumes не коммитятся.
- Через Git LFS хранятся только подготовленные seed-дампы и две
  демонстрационные презентации; после clone выполните `git lfs pull`.
