# OpenRouter для Agentic RAG и демо-проверки

OpenRouter используется только в Agentic RAG, corpus router и reranker. Извлечение графа и
эмбеддинги остаются на Yandex. Внешний Web Search в документальном benchmark выключен.

1. Перевыпустите ключи, которые когда-либо отправлялись в чат.
2. Добавьте новые ключи непосредственно в локальный `.env`:

   ```dotenv
   AGENT_LLM_PROVIDER=openrouter
   OPENROUTER_API_KEYS=<fresh-key-1>,<fresh-key-2>
   OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   OPENROUTER_REASONING=true
   ```

3. Пересоберите backend и worker:

   ```powershell
   docker compose build backend agent-worker
   docker compose up -d backend agent-worker
   ```

4. Запустите возобновляемую проверку. Повтор той же команды с тем же `RunId` продолжит checkpoint:

   ```powershell
   .\run-demo-evaluation.ps1 -Action Run -RunId demo-openrouter-01 -Preflight
   .\run-demo-evaluation.ps1 -Action Status -RunId demo-openrouter-01
   .\run-demo-evaluation.ps1 -Action Render -RunId demo-openrouter-01
   ```

Raw reasoning и API-ключи не сохраняются. В checkpoint находятся только публичный ответ Agentic
RAG, evidence/citations, безопасная диагностика и результаты автоматических проверок.
