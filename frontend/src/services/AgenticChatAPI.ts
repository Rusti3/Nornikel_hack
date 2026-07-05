import { AgenticChatDetails, AgenticTraceEvent, ResponseMode } from '../types';
import { getAgenticRAGEventsUrl, getAgenticRAGJob, startAgenticRAG } from './MEKGAPI';

const terminalStatuses = new Set(['complete', 'failed', 'cancelled']);
const agentEventTypes = [
  'queued',
  'analysis_started',
  'query_analyzed',
  'retrieval_planned',
  'evidence_collected',
  'sufficiency_started',
  'sufficiency_decided',
  'targeted_retry',
  'final_synthesis_started',
  'completed',
  'failed',
  'cancelled',
  'retry_scheduled',
  'cancel_requested',
  'retrieval_preview',
  'retrieval_preview_unavailable',
  'cache_hit',
];

export type AgenticChatOptions = {
  allowExternalWeb: boolean;
  webProfileIds?: string[];
  corpora?: string[];
  geography?: string;
  yearMin?: number;
  yearMax?: number;
  numericMode?: 'boost' | 'strict';
  focusDocumentIds?: string[];
  onUpdate?: (response: ResponseMode) => void;
};

type AgenticJob = {
  job_id: string;
  status: string;
  result?: Record<string, any>;
  state?: Record<string, any>;
  error?: string;
};

const eventLabel = (event: AgenticTraceEvent) => {
  const prefix = event.iteration ? `Итерация ${event.iteration}: ` : '';
  const missing = event.payload?.missing_slots?.length ? ` · не хватает: ${event.payload.missing_slots.join(', ')}` : '';
  const focus = event.payload?.focus?.length ? ` · фокус: ${event.payload.focus.join(', ')}` : '';
  const warnings = event.payload?.warnings?.length ? ` · warnings: ${event.payload.warnings.length}` : '';
  return `${prefix}${event.type}${missing}${focus}${warnings}`;
};

const buildProgressMessage = (job: AgenticJob, trace: AgenticTraceEvent[]) => {
  const lastEvents = trace.slice(-8).map((event) => `- ${eventLabel(event)}`);
  return [
    `### Агент собирает evidence`,
    ``,
    `Статус: \`${job.status || 'queued'}\`${job.job_id ? ` · job \`${job.job_id}\`` : ''}`,
    ``,
    lastEvents.length ? lastEvents.join('\n') : 'Ожидаю worker и первые события trace…',
  ].join('\n');
};

const extractCoverageHint = (result?: Record<string, any>) => {
  const history = result?.state?.search_history;
  if (!Array.isArray(history) || history.length === 0) return undefined;
  const latest = history[history.length - 1];
  return latest?.coverage_hint ?? latest?.retrieval?.coverage_hint;
};

const extractGraphCandidates = (result?: Record<string, any>) => {
  const items = result?.state?.evidence_pack?.items;
  if (!Array.isArray(items)) return [];
  return Array.from(
    new Set(
      items
        .flatMap((item: any) => [
          ...(item?.candidate_entity_ids ?? []),
          ...(item?.graph_candidates ?? []),
          ...(String(item?.source_type ?? '').startsWith('graph_') && item?.source_id ? [item.source_id] : []),
        ])
        .filter((item: any) => typeof item === 'string' && item.trim())
    )
  ) as string[];
};

const toResponseMode = (job: AgenticJob, trace: AgenticTraceEvent[], startedAt: number): ResponseMode => {
  const result = job.result ?? {};
  const previewEvent = [...trace].reverse().find((event) => event.type === 'retrieval_preview');
  const details: AgenticChatDetails = {
    job_id: job.job_id,
    status: job.status,
    mode: result.mode,
    confidence: typeof result.confidence === 'number' ? result.confidence : undefined,
    sources: result.sources ?? [],
    gaps: result.gaps ?? result.state?.evidence_pack?.missing_slots ?? [],
    contradictions: result.contradictions ?? [],
    warnings: [...(result.warnings ?? []), ...(job.error ? [job.error] : [])],
    trace,
    coverage_hint: extractCoverageHint(result),
    graph_candidates: extractGraphCandidates(result),
    diagnostics: {
      searched_iterations: result.searched_iterations,
      llm_available: result.state?.llm_available,
      web_calls: result.state?.web_calls,
    },
    graph_context: result.graph_context,
    preview: previewEvent?.payload?.results ?? [],
  };
  const responseTime = Date.now() - startedAt;
  const isTerminal = terminalStatuses.has(job.status);
  const message =
    result.answer_markdown ||
    (job.status === 'failed'
      ? `### Agentic RAG завершился с ошибкой\n\n${job.error || 'Ошибка не раскрыта backend-ом.'}`
      : job.status === 'cancelled'
        ? '### Agentic RAG отменён\n\nЗапрос остановлен пользователем.'
        : buildProgressMessage(job, trace));
  return {
    message,
    model: 'MEKG Agentic RAG / Yandex Alice',
    response_time: responseTime,
    total_tokens: 0,
    sources: (details.sources ?? []).map((source) => source.url || source.file_name || source.title || source.source_id || '').filter(Boolean),
    error: job.status === 'failed' ? job.error : undefined,
    metric_question: '',
    metric_contexts: '',
    metric_answer: isTerminal ? result.answer_markdown ?? '' : '',
    agentic: details,
  };
};

export const agenticChatAPI = async (question: string, options: AgenticChatOptions) => {
  const startedAt = Date.now();
  const filters: Record<string, any> = {};
  if (options.geography?.trim()) filters.geography = options.geography.trim();
  if (options.yearMin) filters.year_min = options.yearMin;
  if (options.yearMax) filters.year_max = options.yearMax;

  const accepted = await startAgenticRAG({
    query: question,
    allow_external_web: options.allowExternalWeb,
    web_profile_ids: options.allowExternalWeb ? options.webProfileIds ?? [] : [],
    corpora: options.corpora ?? [],
    filters,
    max_iterations: 3,
    include_debug: true,
    focus_document_ids: options.focusDocumentIds ?? [],
  });
  const jobId = accepted.data.job_id;
  let trace: AgenticTraceEvent[] = [];
  const queuedJob: AgenticJob = { job_id: jobId, status: accepted.data.status ?? 'queued' };
  options.onUpdate?.(toResponseMode(queuedJob, trace, startedAt));

  const loadJob = async () => {
    const response = await getAgenticRAGJob(jobId);
    return response.data as AgenticJob;
  };

  const finalJob = await new Promise<AgenticJob>((resolve, reject) => {
    let settled = false;
    let pollTimer: number | undefined;
    const source = new EventSource(getAgenticRAGEventsUrl(jobId));
    const finish = (job: AgenticJob) => {
      if (settled) return;
      settled = true;
      source.close();
      if (pollTimer) window.clearInterval(pollTimer);
      resolve(job);
    };
    const fail = (error: unknown) => {
      if (settled) return;
      settled = true;
      source.close();
      if (pollTimer) window.clearInterval(pollTimer);
      reject(error);
    };
    const poll = async () => {
      try {
        const job = await loadJob();
        options.onUpdate?.(toResponseMode(job, trace, startedAt));
        if (terminalStatuses.has(job.status)) finish(job);
      } catch (error) {
        fail(error);
      }
    };
    pollTimer = window.setInterval(poll, 4000);
    const receive = (raw: Event) => {
      try {
        const event = JSON.parse((raw as MessageEvent).data) as AgenticTraceEvent;
        trace = trace.some((item) => item.id === event.id) ? trace : [...trace, event];
        options.onUpdate?.(toResponseMode({ job_id: jobId, status: 'running' }, trace, startedAt));
        if (['completed', 'failed', 'cancelled'].includes(event.type)) {
          window.setTimeout(() => void poll(), 500);
        }
      } catch (error) {
        fail(error);
      }
    };
    agentEventTypes.forEach((type) => source.addEventListener(type, receive));
    source.onerror = () => {
      source.close();
      void poll();
    };
  });

  const responseMode = toResponseMode(finalJob, trace, startedAt);
  return {
    timeTaken: Date.now() - startedAt,
    response: {
      data: {
        status: 'Success',
        message: responseMode.message,
        error: finalJob.error,
        data: {
          message: responseMode.message,
          info: {
            sources: responseMode.sources ?? [],
            model: responseMode.model,
            total_tokens: 0,
            response_time: responseMode.response_time,
            cypher_query: '',
            context: [],
            entities: responseMode.agentic?.graph_candidates ?? [],
            nodedetails: {},
            error: responseMode.error,
            metric_details: {
              question,
              contexts: '',
              answer: responseMode.metric_answer ?? responseMode.message,
            },
            agentic: responseMode.agentic,
          },
        },
      },
    },
  };
};
