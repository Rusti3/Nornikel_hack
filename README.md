# Nornikel Knowledge Graph / Agentic RAG

## Что открывать после запуска

Главный интерфейс доступен на <http://localhost:8080> и состоит из трёх простых вкладок:

- **Чат** — доказательные ответы Agentic RAG по локальному корпусу; Web Search включается только вручную.
- **Граф** — реальные цепочки `материал → процесс → оборудование → результат`, построенные через узлы `Experiment`. Неполные цепочки показываются с пропущенными звеньями, а не скрываются.
- **Данные** — drag-and-drop загрузка до 10 файлов по 100 МБ. PDF, DOC/DOCX/DOCM, PPTX и XLS/XLSX автоматически проходят text-only parsing, BM25/Postgres, embeddings, извлечение фактов и одну транзакцию Neo4j.

В чат файл можно перетащить прямо вместе с вопросом. Сначала он индексируется, затем вопрос выполняется с приоритетом приложенного документа. У готового ответа доступны экспорт в Markdown/PDF и переход к связанным цепочкам графа.

Дополнительные адреса:

- API и health: <http://localhost:8000/docs>, <http://localhost:8000/health>
- Neo4j Browser: <http://localhost:7474>

## Что нужно установить

- Docker Desktop.
- Git.
- Git LFS: <https://git-lfs.com/>.
- PowerShell 7 или стандартный Windows PowerShell.

## Быстрый запуск

```powershell
git clone https://github.com/Rusti3/Nornikel_hack.git
cd Nornikel_hack
git lfs pull
.\start.ps1
```

При первом запуске `start.ps1` сам создаёт `.env` из готового `.env.example`.
Folder ID, URI моделей и локальные demo-пароли уже заполнены; любые значения
можно изменить в `.env`. Без облачного ключа доступны готовый граф, BM25 и
детерминированный fallback по seed-данным.

Чтобы включить Alice AI, embeddings для новых файлов и Web Search, добавьте
собственный ключ:

```powershell
notepad .env
```

```dotenv
YANDEX_API_KEY=<ваш ключ Yandex AI Studio>
```

Через Git LFS загружаются готовые seed-дампы Postgres и Neo4j.

Скрипт сам загрузит готовые Docker-образы и поднимет стек.

При первом запуске пустые Docker volumes автоматически наполняются готовыми
данными: Postgres/ParadeDB восстанавливает chunks, BM25 и embeddings, Neo4j —
knowledge graph. Повторные запуски существующие volumes не перезаписывают.

Остановка:

```powershell
.\stop.ps1
```

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
