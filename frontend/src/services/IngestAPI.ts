import { getIngestEventsUrl, getIngestJob } from './MEKGAPI';

export type IngestJob = {
  job_id: string;
  status: string;
  state?: Record<string, any>;
  result?: Record<string, any>;
  error?: string;
};

const terminal = new Set(['complete', 'failed', 'cancelled']);
const eventTypes = [
  'queued', 'file_started', 'file_deduplicated', 'parsing', 'chunks_loaded',
  'embedding_progress', 'extracting_first', 'extracting_second', 'neo4j_finalizing',
  'file_completed', 'followup_queued', 'complete', 'complete_with_warnings',
  'retry_scheduled', 'failed', 'cancelled', 'cancel_requested',
];

export const ingestProgressText = (job: IngestJob) => {
  const state = job.state ?? {};
  const event = state.last_event ?? {};
  const stage = state.stage ?? job.status ?? 'queued';
  const file = event.file_name || state.current_file;
  const completed = event.completed;
  const total = event.total;
  return [
    `### Индексация документов`,
    '',
    `Стадия: \`${stage}\`${file ? ` · ${file}` : ''}`,
    typeof completed === 'number' && typeof total === 'number'
      ? `Embeddings: ${completed} / ${total}`
      : 'Документ будет добавлен в BM25, vector search и Neo4j.',
  ].join('\n');
};

export const waitForIngestJob = async (
  jobId: string,
  onUpdate?: (job: IngestJob) => void,
) => new Promise<IngestJob>((resolve, reject) => {
  let settled = false;
  let pollTimer: number | undefined;
  const source = new EventSource(getIngestEventsUrl(jobId));
  const finish = (job: IngestJob) => {
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
      const response = await getIngestJob(jobId);
      const job = response.data as IngestJob;
      onUpdate?.(job);
      if (terminal.has(job.status)) finish(job);
    } catch (error) {
      fail(error);
    }
  };
  pollTimer = window.setInterval(poll, 2500);
  const receive = () => void poll();
  eventTypes.forEach((type) => source.addEventListener(type, receive));
  source.onerror = () => {
    source.close();
    void poll();
  };
  void poll();
});
