1. Главный принцип

Система должна работать не так:

вопрос → поиск → LLM ответ

А так:

вопрос
→ понять, что именно нужно найти
→ составить план поиска
→ собрать evidence
→ проверить sufficient context
→ если не хватает, сделать targeted retry
→ максимум 3 итерации
→ ответить или честно сказать, что данных нет

То есть финальный ответ разрешён только после проверки:

У нас есть достаточно доказательств для ответа?

Если нет — не галлюцинируем.

2. Архитектура agentic RAG
User Query
   ↓
Query Analyzer
   ↓
Retrieval Planner
   ↓
Iteration Loop, max 3:
   ├─ Query Rewriter
   ├─ Retrieval Execution
   ├─ Evidence Pack Builder
   ├─ Rough Draft Generator
   ├─ Sufficient Context Judge
   └─ Stop / Retry Decision
   ↓
Final Synthesis
   или
No Data / Partial Answer

Идеально сделать это как state machine, а не как “LLM сам решает, что делать”.

3. Основные компоненты
1. Query Analyzer

Он превращает пользовательский вопрос в структурированное задание.

Пример запроса:

Какие технические решения организации циркуляции католита при электроэкстракции никеля описаны в мировой практике, и какая скорость потока считается оптимальной?

Analyzer должен выдать:

{
  "intent": "technology_review_with_numeric_recommendation",
  "domain": "hydrometallurgy",
  "main_entities": {
    "process": "nickel electrowinning",
    "object": "catholyte circulation",
    "material": "nickel"
  },
  "required_slots": [
    "technology_solution",
    "flow_velocity_or_flow_rate",
    "unit",
    "effect_or_reason",
    "source",
    "geography"
  ],
  "optional_slots": [
    "equipment_type",
    "industrial_or_lab_scale",
    "limitations",
    "economic_metrics"
  ],
  "filters": {
    "geography": "world_practice",
    "time_range": null,
    "language": ["ru", "en"]
  },
  "answer_type": "comparative_table_plus_summary"
}

Это ключевой момент: не искать просто по тексту вопроса, а понять, какие слоты должны быть заполнены.

2. Retrieval Planner

Planner решает не “какой tool вызвать”, а какую стратегию поиска использовать на этой итерации.

У него должно быть 3 типа стратегий:

Iteration 1: broad search
Найти основные источники, сущности, документы, графовые связи.

Iteration 2: missing-slot search
Искать только то, чего не хватает: числа, единицы, география, baseline, ограничения.

Iteration 3: fallback / relaxed / synonym search
Расширить формулировки, RU/EN синонимы, близкие процессы, соседние термины.

Для металлургии это особенно важно, потому что один и тот же термин может быть:

электроэкстракция никеля
nickel electrowinning
nickel EW
electrolytic extraction of nickel
3. Query Rewriter

Query Rewriter генерирует набор поисковых формулировок под текущую итерацию.

Итерация 1 — широкий поиск
{
  "iteration": 1,
  "goal": "find candidate evidence",
  "queries": [
    "циркуляция католита электроэкстракция никеля скорость потока",
    "nickel electrowinning catholyte circulation flow velocity",
    "nickel electrowinning electrolyte flow rate cathode compartment",
    "technical solutions catholyte circulation nickel electroextraction"
  ]
}
Итерация 2 — поиск недостающих слотов

Если после первой итерации не хватает скорости потока:

{
  "iteration": 2,
  "goal": "fill missing numeric flow velocity",
  "queries": [
    "nickel electrowinning catholyte flow velocity m/s",
    "nickel electrowinning electrolyte circulation rate optimal",
    "скорость циркуляции электролита электроэкстракция никеля м/с",
    "catholyte flow rate nickel electrowinning optimal range"
  ]
}
Итерация 3 — fallback
{
  "iteration": 3,
  "goal": "relaxed search with synonyms and adjacent terms",
  "queries": [
    "nickel electrorefining electrolyte circulation flow rate",
    "nickel electrowinning mass transfer catholyte velocity",
    "electrolyte hydrodynamics nickel electrodeposition",
    "circulation of electrolyte in nickel electrodeposition cells"
  ]
}

Важно: третья итерация может расширять поиск, но не должна уходить в другую тему.

4. Evidence Pack Builder

После каждого поиска ты не должен сразу отдавать chunks в LLM. Сначала собери Evidence Pack.

{
  "question": "...",
  "iteration": 1,
  "evidence_items": [
    {
      "id": "ev_001",
      "source_id": "doc_12",
      "source_type": "article",
      "title": "...",
      "year": 2021,
      "geography": "foreign",
      "snippet": "...",
      "extracted_facts": [
        {
          "type": "numeric_parameter",
          "name": "flow_velocity",
          "value_min": 0.03,
          "value_max": 0.08,
          "unit": "m/s"
        }
      ],
      "supports_slots": [
        "flow_velocity_or_flow_rate",
        "unit",
        "source"
      ],
      "confidence": 0.78
    }
  ],
  "covered_slots": [
    "technology_solution",
    "source"
  ],
  "missing_slots": [
    "flow_velocity_or_flow_rate",
    "effect_or_reason",
    "geography"
  ],
  "contradictions": []
}

То есть Evidence Pack — это не просто список текстов. Это структурированная “папка доказательств”.

5. Rough Draft Generator

Это внутренний черновик, не показываем пользователю.

Зачем он нужен: Sufficient Context Judge должен понять, может ли система уже написать нормальный ответ. Google в своей схеме sufficient context agent проверяет не только retrieved snippets, но и intermediate draft, а также делает missing pieces analysis.

Черновик может быть плохим, это нормально. Его задача — выявить дыры.

Например rough draft:

Найдены три решения циркуляции католита: принудительная циркуляция насосом,
подача через распределительный коллектор и циркуляция за счёт гидродинамики ванны.
Однако точные скорости потока указаны только в одном источнике...

После этого Judge говорит:

Не хватает числового диапазона скорости по большинству решений.
Нужен допоиск по flow velocity / flow rate.
6. Sufficient Context Judge

Это центральный компонент.

Он получает:

1. Исходный вопрос
2. Required slots
3. Evidence Pack
4. Rough Draft
5. Историю прошлых поисков

И возвращает строгий JSON.

{
  "sufficient": false,
  "score": 62,
  "answer_mode": "search_more",
  "covered_slots": {
    "technology_solution": "covered",
    "flow_velocity_or_flow_rate": "partial",
    "unit": "partial",
    "effect_or_reason": "missing",
    "source": "covered",
    "geography": "partial"
  },
  "critical_missing": [
    "effect_or_reason",
    "flow_velocity_or_flow_rate"
  ],
  "contradictions": [],
  "reason": "Найдены технические решения, но численные скорости потока представлены неполно, а обоснование оптимальности отсутствует.",
  "next_search_focus": [
    "optimal catholyte flow velocity",
    "mass transfer effect",
    "industrial nickel electrowinning electrolyte circulation"
  ],
  "can_answer_partially": true
}

Контекст считается достаточным только если:

1. Все critical slots закрыты.
2. Есть source для каждого важного утверждения.
3. Числа имеют единицы измерения.
4. Нет неразрешённых противоречий.
5. Ответ не требует данных, которых нет в evidence.

Google отдельно подчёркивает, что релевантность контекста недостаточна: контекст может быть похож на вопрос, но не содержать ответа. Поэтому нужен именно sufficiency check перед генерацией.

7. Scoring достаточного контекста

Я бы сделал гибрид: жёсткие правила + LLM judge.

Hard gates

Если вопрос требует численное значение, то без числа и единицы answer запрещён.

if question.requires_numeric_answer:
    if not evidence.has_numeric_value or not evidence.has_unit:
        sufficient = False

Если вопрос требует сравнение РФ vs зарубежная практика, без geography answer запрещён.

if question.requires_geography_comparison:
    if not evidence.has_domestic_or_foreign_labels:
        sufficient = False

Если вопрос требует “за последние 5 лет”, без года источника answer запрещён.

if question.requires_time_filter:
    if not evidence.has_year:
        sufficient = False
Soft score
100 баллов максимум:

25 — закрыты все critical slots
20 — есть численные значения и единицы
15 — есть прямые источники для claims
15 — несколько независимых источников
10 — есть география / год / тип практики
10 — нет противоречий или они объяснены
5  — данные хорошо структурированы для таблицы

Порог:

80–100: можно давать полный ответ
60–79: можно partial answer + явно указать gaps
0–59: искать дальше, если итерации остались

Но важнее soft score — hard gates. Если нет обязательного числового параметра, score не должен спасать ответ.

8. Stop / Retry Decision

После Judge есть только 4 варианта.

{
  "action": "answer_full"
}

Когда всё найдено.

{
  "action": "search_more",
  "focus": ["missing numeric range", "foreign practice"]
}

Когда итерации ещё есть.

{
  "action": "answer_partial"
}

Когда есть полезные данные, но не всё найдено, а итерации закончились.

{
  "action": "no_data"
}

Когда в доступных источниках вообще нет достаточных данных.

9. Максимум 3 итерации

Вот прям так:

MAX_ITERATIONS = 3

for iteration in range(1, MAX_ITERATIONS + 1):
    queries = rewrite_queries(question, state, iteration)
    raw_results = execute_retrieval(queries)
    evidence_pack = build_evidence_pack(raw_results, state)
    rough_draft = generate_rough_draft(question, evidence_pack)

    verdict = sufficient_context_judge(
        question=question,
        required_slots=state.required_slots,
        evidence_pack=evidence_pack,
        rough_draft=rough_draft,
        search_history=state.search_history
    )

    state.update(evidence_pack, verdict)

    if verdict.action == "answer_full":
        return synthesize_final_answer(state)

    if iteration < MAX_ITERATIONS and verdict.action == "search_more":
        continue

    if verdict.can_answer_partially:
        return synthesize_partial_answer(state)

    return no_data_answer(state)
10. Что происходит на каждой итерации
Итерация 1 — broad retrieval

Цель: найти основные документы и сущности.

Ищем широко:
- основные методы
- основные материалы
- основные процессы
- обзорные источники
- графовые связи

Результат:

Есть общая картина, но часто нет чисел.
Итерация 2 — missing-slot retrieval

Цель: закрыть конкретные дыры.

Пример missing slots:

нет скорости потока
нет CAPEX/OPEX
нет российской практики
нет значения температуры
нет baseline

Ищем уже не общий вопрос, а конкретные недостающие факты.

"catholyte flow velocity m/s"
"капитальные затраты закачка шахтных вод"
"heap leaching cold climate metal recovery"
Итерация 3 — fallback retrieval

Цель: последняя попытка найти через синонимы, соседние формулировки и ослабление фильтров.

- русский ↔ английский
- термин ↔ аббревиатура
- узкий процесс ↔ соседний процесс
- точная фраза ↔ более общий механизм

Например:

электроэкстракция никеля
→ nickel electrowinning
→ nickel electrodeposition
→ nickel electrorefining
→ electrolyte hydrodynamics

Но нельзя уходить слишком далеко. Для этого нужен drift guard.

11. Drift Guard

Drift Guard проверяет, что новая итерация не ушла в другую тему.

{
  "original_core": ["nickel", "electrowinning", "catholyte circulation"],
  "new_query": "copper electrorefining electrolyte hydrodynamics",
  "drift_risk": "medium",
  "allowed": true,
  "reason": "Copper electrorefining is adjacent electrochemical process, allowed only for fallback analogies, not as primary evidence."
}

Правило:

Primary answer можно строить только по прямым источникам.
Смежные источники можно использовать только как analogies / recommendations.

Это важно для металлургии. Иначе система начнёт переносить выводы из меди на никель как будто это одно и то же.

12. State object всей системы

Сделай один объект состояния, который проходит через весь agentic loop.

{
  "query_id": "q_123",
  "original_query": "...",
  "parsed_query": {
    "intent": "...",
    "entities": {},
    "constraints": {},
    "required_slots": [],
    "optional_slots": []
  },
  "iteration": 1,
  "search_history": [
    {
      "iteration": 1,
      "queries": [],
      "results_count": 24,
      "useful_results_count": 7
    }
  ],
  "evidence_pack": {
    "items": [],
    "claims": [],
    "numeric_facts": [],
    "graph_paths": [],
    "contradictions": [],
    "gaps": []
  },
  "sufficiency": {
    "score": 0,
    "sufficient": false,
    "missing_slots": [],
    "action": "search_more"
  },
  "final_mode": null
}

Это даёт тебе контроль, дебаг и красивый trace в UI.

13. Типы answer mode

Финальный ответ должен иметь режим.

Full answer
{
  "mode": "full_answer",
  "reason": "All critical slots are covered."
}

Ответ обычный: вывод, таблица, источники.

Partial answer
{
  "mode": "partial_answer_with_gaps",
  "reason": "Some relevant evidence found, but critical numeric values are incomplete."
}

Формулировка:

В доступных источниках найдены технические решения A, B, C.
Однако точный оптимальный диапазон скорости потока подтверждён только для A.
По B и C данных о скорости не найдено. Поэтому вывод по оптимальной скорости является частичным.
No data
{
  "mode": "no_data",
  "reason": "After 3 iterations no evidence was found that directly answers the query."
}

Формулировка:

В доступном корпусе не найдено данных, которые напрямую отвечают на вопрос.
Были проверены: ...
Не найдено: ...
Ближайшие найденные материалы: ...
Рекомендуемые следующие источники: ...

Важно: лучше писать “в доступном корпусе / доступных источниках системы”, а не “нигде в мире”, если у тебя нет web/patent/external search. “Нигде” можно говорить только если у тебя реально подключены внешние базы и ты их тоже проверил.

14. Prompt для Query Analyzer
Ты анализируешь технический R&D запрос в горно-металлургической области.

Твоя задача:
1. Определи intent запроса.
2. Извлеки материалы, процессы, условия, географию, временной диапазон.
3. Определи обязательные слоты, без которых нельзя дать корректный ответ.
4. Определи опциональные слоты.
5. Определи, нужен ли числовой ответ.
6. Определи, нужен ли сравнительный анализ.
7. Верни только JSON.

Запрос:
{user_query}

JSON schema:
{
  "intent": "...",
  "domain": "...",
  "entities": {
    "materials": [],
    "processes": [],
    "equipment": [],
    "properties": [],
    "substances": []
  },
  "constraints": {
    "numeric": [],
    "geography": null,
    "time_range": null
  },
  "required_slots": [],
  "optional_slots": [],
  "requires_numeric_answer": true/false,
  "requires_geography_comparison": true/false,
  "answer_type": "table|review|experiment_list|recommendation|gap_analysis"
}
15. Prompt для Sufficient Context Judge
Ты проверяешь, достаточно ли evidence для ответа на технический вопрос.
Не отвечай на вопрос. Только оцени достаточность.

Контекст считается достаточным, если:
- покрыты все обязательные слоты;
- для числовых утверждений есть значения и единицы;
- для каждого важного вывода есть источник;
- противоречия либо отсутствуют, либо явно объяснимы;
- evidence позволяет дать ответ без догадок.

Исходный вопрос:
{question}

Обязательные слоты:
{required_slots}

Evidence Pack:
{evidence_pack}

Черновой ответ:
{rough_draft}

История поиска:
{search_history}

Верни JSON:
{
  "sufficient": true/false,
  "score": 0-100,
  "action": "answer_full|search_more|answer_partial|no_data",
  "covered_slots": {},
  "missing_slots": [],
  "critical_missing": [],
  "contradictions": [],
  "reason": "...",
  "next_search_focus": [],
  "can_answer_partially": true/false
}

Главное правило: Judge не должен быть оптимистичным. Он должен быть “инженером-проверяющим”.

16. Prompt для Final Synthesis
Ты формируешь финальный ответ только на основе Evidence Pack.
Запрещено добавлять факты, которых нет в Evidence Pack.
Если данных недостаточно, явно укажи это.
Все численные значения должны иметь единицы.
Каждое важное утверждение должно ссылаться на source_id.
Если есть противоречия, покажи их отдельно.
Если есть gaps, покажи их отдельно.

Вопрос:
{question}

Evidence Pack:
{evidence_pack}

Sufficiency verdict:
{sufficiency}

Сформируй ответ в структуре:
1. Краткий вывод
2. Таблица найденных решений/экспериментов
3. Подтверждающие источники
4. Противоречия
5. Пробелы в данных
6. Уровень уверенности
17. Что показывать в UI как agent trace

Это будет сильно смотреться на демо.

Понимание запроса:
- Процесс: электроэкстракция никеля
- Объект: циркуляция католита
- Требуется: технические решения + скорость потока
- География: мировая практика

Итерация 1:
- Найдено 18 источников
- Полезных: 6
- Покрыто: technical solutions, sources
- Не хватает: flow velocity, effect

Итерация 2:
- Поиск: flow velocity / electrolyte circulation rate
- Найдено 7 источников
- Полезных: 3
- Покрыто: flow velocity partially

Итерация 3:
- Поиск по синонимам: electrodeposition / hydrodynamics
- Найдено 4 источника
- Полезных: 1

Вердикт:
- Контекст частично достаточен
- Полный ответ по оптимальной скорости невозможен
- Показаны gaps

Жюри увидит, что система не просто “поговорила”, а реально провела исследовательский процесс.

18. Как формировать no-data ответ

После 3 итераций нельзя просто сказать:

Нет данных.

Надо сказать умно:

В доступном корпусе не найдено данных, которые напрямую подтверждают оптимальную скорость циркуляции католита при электроэкстракции никеля.

Что было проверено:
1. Прямые запросы по “электроэкстракция никеля / catholyte circulation”.
2. Английские синонимы “nickel electrowinning / electrolyte circulation / flow velocity”.
3. Смежные термины “nickel electrodeposition / electrolyte hydrodynamics”.

Что найдено:
- Найдены описания схем циркуляции.
- Найдены упоминания влияния перемешивания на массоперенос.
- Не найдены подтверждённые числовые диапазоны оптимальной скорости потока.

Пробел в знаниях:
Нет источников с комбинацией:
nickel electrowinning + catholyte circulation + optimal flow velocity + industrial validation.

Рекомендация:
Проверить патентные документы, регламенты проектирования ванн электроэкстракции и внутренние протоколы опытов по гидродинамике электролита.

Вот это выглядит как настоящий исследовательский ассистент.

19. Как делать “нет таких данных нигде”

Технически правильно разделить 3 уровня:

NO_DIRECT_DATA
Прямых данных по вопросу нет, но есть смежные.

NO_NUMERIC_DATA
Описание есть, численных параметров нет.

NO_EVIDENCE_FOUND
Вообще не найдено релевантных источников.

OUT_OF_SCOPE
Вопрос требует данных, которых нет в схеме/корпусе.

Пример JSON:

{
  "final_mode": "NO_NUMERIC_DATA",
  "message": "Найдены качественные описания решений, но не найден подтверждённый числовой диапазон скорости потока.",
  "searched_iterations": 3,
  "missing_slots": ["flow_velocity_or_flow_rate", "unit"],
  "nearest_evidence": ["ev_001", "ev_007"]
}
20. Главная логика достаточности для ваших типов вопросов
Тип 1: “Какие методы подходят?”

Обязательные слоты:

method
applicability_condition
input_constraints
output_requirement
source
limitations

Если нет applicability condition — нельзя рекомендовать метод.

Тип 2: “Какая скорость/температура/концентрация оптимальна?”

Обязательные слоты:

parameter_name
numeric_value_or_range
unit
process
material/system
source
basis_for_optimality

Если есть число, но нет basis_for_optimality, надо писать:

“указанный диапазон встречается в источниках”, а не “оптимальный”.
Тип 3: “Покажи все эксперименты за последние 5 лет”

Обязательные слоты:

experiment_id_or_source
year
material
process
result_or_property
source

Если нет year — нельзя уверенно включать в “последние 5 лет”.

Тип 4: “РФ vs зарубежная практика”

Обязательные слоты:

technology
geography
source
practice_type
parameter_or_result

Если нет geography — нельзя класть источник в РФ/зарубежье.

Тип 5: “Где пробелы?”

Обязательные слоты:

target_combination
searched_entities
missing_edges_or_missing_slots
nearest_related_evidence

Gap answer можно давать даже при insufficient context, потому что сам факт отсутствия связи после поиска — полезный результат.

21. Не делай “свободного агента”

Плохая архитектура:

LLM думает → сам вызывает что хочет → сам решает когда ответить

Хорошая архитектура:

LLM заполняет строго заданные JSON-контракты.
State machine решает, можно ли двигаться дальше.

То есть:

Analyzer — JSON
Planner — JSON
Judge — JSON
Synthesis — Markdown/JSON

Так ты избежишь хаоса.

22. Финальная схема
                 ┌──────────────────┐
                 │   User Query      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Query Analyzer    │
                 │ intent + slots    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Retrieval Planner │
                 │ strategy          │
                 └────────┬─────────┘
                          ↓
        ┌────────────────────────────────────┐
        │ Iterative Evidence Loop, max 3      │
        │                                    │
        │  Query Rewriter                    │
        │       ↓                            │
        │  Retrieval Execution               │
        │       ↓                            │
        │  Evidence Pack Builder             │
        │       ↓                            │
        │  Rough Draft                       │
        │       ↓                            │
        │  Sufficient Context Judge          │
        │       ↓                            │
        │  answer / retry / partial / no data│
        └────────────────────────────────────┘
                          ↓
              ┌───────────────────────┐
              │ Final Synthesis        │
              │ or No Data Response    │
              └───────────────────────┘