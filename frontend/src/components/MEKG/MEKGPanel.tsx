import { Button, Dialog, Flex, StatusIndicator, Typography } from '@neo4j-ndl/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  cancelAgenticRAGJob,
  crossCorpusSearch,
  getAgenticRAGEventsUrl,
  getAgenticRAGJob,
  getMEKGExportUrl,
  getMEKGProfile,
  getMEKGQA,
  getMEKGReadiness,
  getMEKGReviewQueue,
  getWebSearchProfiles,
  initializeMEKG,
  reviewMEKGFact,
  runMEKGQA,
  startAgenticRAG,
} from '../../services/MEKGAPI';
import { showErrorToast, showSuccessToast } from '../../utils/Toasts';

type Props = { open: boolean; onClose: () => void };
type AgentEvent = { id: number; type: string; iteration?: number; payload: Record<string, any>; created_at: string };
const cellStyle = { padding: '6px 10px', borderBottom: '1px solid #d7d9dc' };
const terminalStatuses = new Set(['complete', 'failed', 'cancelled']);
const agentEventTypes = [
  'queued', 'analysis_started', 'query_analyzed', 'retrieval_planned', 'evidence_collected',
  'sufficiency_started', 'sufficiency_decided', 'targeted_retry', 'final_synthesis_started',
  'completed', 'failed', 'cancelled', 'retry_scheduled',
];

export default function MEKGPanel({ open, onClose }: Props) {
  const [profile, setProfile] = useState(localStorage.getItem('extractionProfile') ?? 'mekg');
  const [profileInfo, setProfileInfo] = useState<any>(null);
  const [qa, setQA] = useState<any>(null);
  const [readiness, setReadiness] = useState<any>(null);
  const [queue, setQueue] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchSlots, setSearchSlots] = useState('');
  const [searchResults, setSearchResults] = useState<any>(null);
  const [webProfiles, setWebProfiles] = useState<any[]>([]);
  const [selectedWebProfiles, setSelectedWebProfiles] = useState<string[]>(['journals', 'mining_metals']);
  const [agentQuery, setAgentQuery] = useState('');
  const [allowExternalWeb, setAllowExternalWeb] = useState(false);
  const [agentJob, setAgentJob] = useState<any>(null);
  const [agentTrace, setAgentTrace] = useState<AgentEvent[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [profileResponse, qaResponse, queueResponse, webResponse, readinessResponse] = await Promise.all([
        getMEKGProfile(),
        getMEKGQA(),
        getMEKGReviewQueue(),
        getWebSearchProfiles(),
        getMEKGReadiness().catch((error) => ({ data: { warnings: [error instanceof Error ? error.message : 'Readiness endpoint unavailable'] } })),
      ]);
      setProfileInfo(profileResponse.data);
      setQA(qaResponse.data);
      setQueue(queueResponse.data.items ?? []);
      setWebProfiles(webResponse.data.profiles ?? []);
      setReadiness(readinessResponse.data);
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to load MEKG status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { if (open) refresh(); }, [open, refresh]);
  useEffect(() => () => eventSourceRef.current?.close(), []);

  const selectProfile = (value: string) => {
    localStorage.setItem('extractionProfile', value);
    setProfile(value);
    showSuccessToast(value === 'mekg' ? 'MEKG evidence profile enabled' : 'Generic profile enabled');
  };

  const initialize = async () => {
    setLoading(true);
    try {
      await initializeMEKG();
      showSuccessToast('MEKG ontology and constraints initialized');
      await refresh();
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'MEKG initialization failed');
    } finally { setLoading(false); }
  };

  const review = async (id: string, status: 'expert_validated' | 'rejected') => {
    try { await reviewMEKGFact(id, status); await refresh(); }
    catch (error) { showErrorToast(error instanceof Error ? error.message : 'Review failed'); }
  };

  const search = async () => {
    if (searchQuery.trim().length < 2) return;
    setLoading(true);
    try {
      const response = await crossCorpusSearch({
        query: searchQuery.trim(), intent: 'research',
        target_slots: searchSlots.split(',').map((item) => item.trim()).filter(Boolean), final_k: 20,
      });
      setSearchResults(response.data);
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Cross-corpus search failed');
    } finally { setLoading(false); }
  };

  const loadAgentJob = useCallback(async (jobId: string) => {
    try {
      const response = await getAgenticRAGJob(jobId);
      setAgentJob(response.data);
      if (terminalStatuses.has(response.data.status)) eventSourceRef.current?.close();
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to load agent job');
    }
  }, []);

  const connectAgentEvents = useCallback((jobId: string) => {
    eventSourceRef.current?.close();
    const source = new EventSource(getAgenticRAGEventsUrl(jobId));
    eventSourceRef.current = source;
    const receive = (raw: Event) => {
      const event = JSON.parse((raw as MessageEvent).data) as AgentEvent;
      setAgentTrace((current) => current.some((item) => item.id === event.id) ? current : [...current, event]);
      if (['completed', 'failed', 'cancelled'].includes(event.type)) {
        window.setTimeout(() => loadAgentJob(jobId), 700);
      }
    };
    agentEventTypes.forEach((type) => source.addEventListener(type, receive));
    source.onerror = () => { source.close(); void loadAgentJob(jobId); };
  }, [loadAgentJob]);

  const startAgent = async () => {
    if (agentQuery.trim().length < 2) return;
    setLoading(true);
    setAgentTrace([]);
    setAgentJob(null);
    try {
      const response = await startAgenticRAG({
        query: agentQuery.trim(),
        allow_external_web: allowExternalWeb,
        web_profile_ids: allowExternalWeb ? selectedWebProfiles : [],
        max_iterations: 3,
        include_debug: true,
      });
      setAgentJob(response.data);
      connectAgentEvents(response.data.job_id);
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to start Agentic RAG');
    } finally { setLoading(false); }
  };

  const cancelAgent = async () => {
    if (!agentJob?.job_id) return;
    try {
      await cancelAgenticRAGJob(agentJob.job_id);
      await loadAgentJob(agentJob.job_id);
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to cancel agent job');
    }
  };

  const toggleWebProfile = (id: string) => {
    setSelectedWebProfiles((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      if (current.length >= 2) return [current[1], id];
      return [...current, id];
    });
  };

  const runQA = async () => {
    setLoading(true);
    try {
      const response = await runMEKGQA();
      setQA((current: any) => ({ ...(current ?? {}), pending: true, job_id: response.data.job_id }));
      showSuccessToast('Graph QA queued in the background');
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to queue graph QA');
    } finally { setLoading(false); }
  };

  return (
    <Dialog isOpen={open} size='large' onClose={onClose} hasDisabledCloseButton={false}>
      <Dialog.Header><Typography variant='h3'>Metallurgical Evidence Knowledge Graph</Typography></Dialog.Header>
      <Dialog.Content className='overflow-y-auto' style={{ maxHeight: '76vh', maxWidth: '100%' }}>
        <Flex justifyContent='space-between' alignItems='center' flexWrap='wrap' gap='2' className='mb-4'>
          <Flex gap='2' alignItems='center'>
            <Typography variant='body-medium'>Extraction profile:</Typography>
            <Button fill={profile === 'mekg' ? 'filled' : 'outlined'} onClick={() => selectProfile('mekg')}>MEKG evidence</Button>
            <Button fill={profile === 'generic' ? 'filled' : 'outlined'} onClick={() => selectProfile('generic')}>Generic</Button>
          </Flex>
          <Flex gap='2'>
            <Button onClick={initialize} isLoading={loading}>Initialize ontology</Button>
            <Button onClick={refresh} isLoading={loading}>Refresh status</Button>
          </Flex>
        </Flex>
        {profileInfo && (
          <div className='n-bg-palette-neutral-bg-weak p-3 rounded mb-4'>
            <Typography variant='body-medium'>Alice: {profileInfo.llm_model} · Vision: {profileInfo.vision_model} · Embeddings: {profileInfo.embedding_dimensions}D</Typography>
            <Typography variant='body-small'>Yandex data logging: {String(profileInfo.data_logging)}</Typography>
          </div>
        )}

        <Typography variant='h4'>Agentic RAG</Typography>
        <div className='n-bg-palette-neutral-bg-weak p-3 rounded mb-5' data-testid='agentic-rag-panel'>
          <textarea
            data-testid='agentic-rag-query'
            className='w-full p-2 mb-2 rounded border min-h-24'
            value={agentQuery}
            onChange={(event) => setAgentQuery(event.target.value)}
            placeholder='Технический вопрос: методы, параметры, сравнение практик или пробелы в данных'
          />
          <label className='flex gap-2 items-start mb-2'>
            <input
              data-testid='allow-external-web'
              type='checkbox'
              checked={allowExternalWeb}
              onChange={(event) => setAllowExternalWeb(event.target.checked)}
            />
            <span>
              Разрешить внешний Yandex Web Search. Во внешний контур уйдут только очищенные поисковые формулировки;
              внутренние чанки и пути не передаются.
            </span>
          </label>
          {allowExternalWeb && <div className='mb-3'>
            <Typography variant='body-small'>Профили источников — максимум два за итерацию:</Typography>
            <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1 mt-1'>
              {webProfiles.map((item: any) => <label key={item.id} className='flex gap-1 items-center'>
                <input
                  type='checkbox'
                  checked={selectedWebProfiles.includes(item.id)}
                  onChange={() => toggleWebProfile(item.id)}
                />
                <span title={item.description}>{item.title}{item.metadata_only ? ' (metadata)' : ''}</span>
              </label>)}
            </div>
          </div>}
          <Flex gap='2'>
            <Button
              data-testid='start-agentic-rag'
              onClick={startAgent}
              isLoading={loading}
              isDisabled={agentQuery.trim().length < 2 || (allowExternalWeb && selectedWebProfiles.length === 0)}
            >Start evidence loop</Button>
            {agentJob && !terminalStatuses.has(agentJob.status) &&
              <Button fill='outlined' onClick={cancelAgent}>Cancel</Button>}
          </Flex>
          {agentJob && <div className='mt-3'>
            <Typography variant='body-medium'>Job {agentJob.job_id} · {agentJob.status}</Typography>
            <div className='mt-2 max-h-48 overflow-y-auto bg-white rounded p-2' data-testid='agent-trace'>
              {agentTrace.map((event) => <div key={event.id} className='text-xs mb-1'>
                <strong>{event.iteration ? `Iteration ${event.iteration}: ` : ''}{event.type}</strong>
                {event.payload?.missing_slots && ` · missing: ${event.payload.missing_slots.join(', ')}`}
                {event.payload?.focus && ` · focus: ${event.payload.focus.join(', ')}`}
              </div>)}
              {agentTrace.length === 0 && <span className='text-xs'>Waiting for worker…</span>}
            </div>
          </div>}
          {agentJob?.result && <div className='mt-3 bg-white rounded p-3' data-testid='agent-result'>
            <Typography variant='h5'>{agentJob.result.mode} · confidence {Number(agentJob.result.confidence).toFixed(2)}</Typography>
            <div className='prose max-w-none mt-2'><ReactMarkdown>{agentJob.result.answer_markdown}</ReactMarkdown></div>
            {(agentJob.result.sources ?? []).filter((item: any) => item.url).slice(0, 20).map((item: any) =>
              <div key={item.label} className='text-xs mt-1'>
                [{item.label}] <a href={item.url} target='_blank' rel='noreferrer'>{item.title || item.url}</a>
                {item.metadata_only ? ' · metadata only' : ''}
              </div>)}
            {(agentJob.result.warnings ?? []).length > 0 &&
              <pre className='mt-2 text-xs whitespace-pre-wrap'>{agentJob.result.warnings.join('\n')}</pre>}
          </div>}
        </div>

        <Typography variant='h5'>Cross-corpus evidence search</Typography>
        <div className='n-bg-palette-neutral-bg-weak p-3 rounded mb-4'>
          <input className='w-full p-2 mb-2 rounded border' value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)} placeholder='Research question or evidence query' />
          <input className='w-full p-2 mb-2 rounded border' value={searchSlots}
            onChange={(event) => setSearchSlots(event.target.value)}
            placeholder='Target slots, comma separated (material, process, metric)' />
          <Button onClick={search} isLoading={loading}>Search all corpora</Button>
          {searchResults?.data?.coverage_hint && <pre className='mt-2 text-xs whitespace-pre-wrap'>{JSON.stringify(searchResults.data.coverage_hint, null, 2)}</pre>}
          {(searchResults?.data?.results ?? []).slice(0, 20).map((item: any) => (
            <div key={item.chunk_id} className='mt-2 p-2 bg-white rounded'>
              <Typography variant='body-medium'>{item.file_name} · score {Number(item.score).toFixed(3)}</Typography>
              <Typography variant='body-small'>{String(item.text ?? '').slice(0, 600)}</Typography>
              <small>{item.corpus_id} · page {item.page_number ?? '—'} · {item.chunk_id}</small>
            </div>
          ))}
        </div>

        <Typography variant='h5'>Demo+QA readiness for “Научный клубок”</Typography>
        <div className='n-bg-palette-neutral-bg-weak p-3 rounded mb-4' data-testid='mekg-readiness-dashboard'>
          {readiness ? <>
            {(readiness.warnings ?? []).length > 0 &&
              <pre className='mb-3 text-xs whitespace-pre-wrap bg-white rounded p-2'>{readiness.warnings.join('\n')}</pre>}
            <div className='grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mb-3'>
              <div className='bg-white rounded p-2'>
                <Typography variant='body-medium'>Postgres pipeline</Typography>
                <table className='w-full text-xs table-fixed break-words'><tbody>
                  <tr><td style={cellStyle}>run</td><td style={cellStyle}>{readiness.pipeline?.run_id ?? '—'}</td></tr>
                  <tr><td style={cellStyle}>chunks</td><td style={cellStyle}>{readiness.pipeline?.chunks ?? 0}</td></tr>
                  <tr><td style={cellStyle}>embedded</td><td style={cellStyle}>{readiness.pipeline?.embedded ?? 0}</td></tr>
                  {Object.entries(readiness.pipeline?.stages ?? {}).map(([stage, value]) =>
                    <tr key={stage}><td style={cellStyle}>{stage}</td><td style={cellStyle}>{String(value)}</td></tr>)}
                </tbody></table>
              </div>
              <div className='bg-white rounded p-2'>
                <Typography variant='body-medium'>Neo4j graph</Typography>
                <table className='w-full text-xs table-fixed break-words'><tbody>
                  {Object.entries(readiness.graph?.metrics ?? {}).map(([key, value]) =>
                    <tr key={key}><td style={cellStyle}>{key}</td><td style={cellStyle}>{String(value)}</td></tr>)}
                </tbody></table>
              </div>
              <div className='bg-white rounded p-2'>
                <Typography variant='body-medium'>Coverage by corpus/category</Typography>
                <table className='w-full text-xs table-fixed break-words'><tbody>
                  {(readiness.graph?.coverage ?? []).map((row: any) =>
                    <tr key={row.category}><td style={cellStyle}>{row.category}</td><td style={cellStyle}>{row.documents}</td></tr>)}
                </tbody></table>
              </div>
            </div>
            <div className='grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mb-3'>
              {Object.entries(readiness.graph?.top ?? {}).map(([kind, rows]) =>
                <div key={kind} className='bg-white rounded p-2'>
                  <Typography variant='body-medium'>{kind}</Typography>
                  {(rows as any[]).length === 0 && <Typography variant='body-small'>No data yet</Typography>}
                  {(rows as any[]).map((row) =>
                    <div key={`${kind}-${row.name}`} className='text-xs'>{row.name ?? '—'} · {row.links ?? 0}</div>)}
                </div>)}
            </div>
            <table className='w-full table-fixed break-words text-xs bg-white rounded'><thead><tr><th style={cellStyle}>Requirement</th><th style={cellStyle}>Status</th><th style={cellStyle}>Evidence</th></tr></thead><tbody>
              {(readiness.checklist ?? []).map((item: any) => <tr key={item.id}>
                <td style={cellStyle}>{item.title}</td>
                <td style={cellStyle}>{item.status}</td>
                <td style={cellStyle}>{item.evidence}</td>
              </tr>)}
            </tbody></table>
          </> : <Typography variant='body-small'>Readiness snapshot has not been loaded.</Typography>}
        </div>

        <Flex justifyContent='space-between' alignItems='center'>
          <Typography variant='h5'>Verified graph quality</Typography>
          <Button size='small' fill='outlined' onClick={runQA} isDisabled={Boolean(qa?.pending)}>Run QA in background</Button>
        </Flex>
        {qa ? <>
          <Flex gap='2' alignItems='center' className='my-2'>
            <StatusIndicator type={qa.passed ? 'success' : (qa.pending ? 'warning' : 'danger')} />
            <Typography variant='body-medium'>
              {qa.pending ? `QA queued (${qa.job_id})` : (qa.passed ? 'All quality gates pass' : 'Quality snapshot unavailable or violations found')}
            </Typography>
          </Flex>
          <table className='w-full table-fixed break-words mb-4'><tbody>
            {Object.entries(qa.metrics ?? {}).map(([key, value]) => <tr key={key}><td style={cellStyle}>{key}</td><td style={cellStyle}>{String(value)}</td></tr>)}
            <tr><td style={cellStyle}>SHACL violations</td><td style={cellStyle}>{qa.shacl?.violations ?? '—'}</td></tr>
          </tbody></table>
        </> : <Typography variant='body-small'>QA has not been loaded.</Typography>}
        <Flex justifyContent='space-between' alignItems='center' className='mb-2'>
          <Typography variant='h5'>Expert review queue ({queue.length})</Typography>
          <Flex gap='2'><a href={getMEKGExportUrl('turtle')} target='_blank' rel='noreferrer'>Turtle</a><a href={getMEKGExportUrl('jsonld')} target='_blank' rel='noreferrer'>JSON-LD</a></Flex>
        </Flex>
        <table className='w-full table-fixed break-words text-xs'><thead><tr><th style={cellStyle}>Item</th><th style={cellStyle}>Reason/status</th><th style={cellStyle}>Decision</th></tr></thead><tbody>
          {queue.slice(0, 50).map((item) => <tr key={item.id}>
            <td style={cellStyle}><div>{item.title || item.labels?.join(', ')}</div><small>{item.id}</small></td>
            <td style={cellStyle}>{item.reason || item.status}</td>
            <td style={cellStyle}><Flex gap='1'><Button size='small' onClick={() => review(item.id, 'expert_validated')}>Approve</Button><Button size='small' fill='outlined' onClick={() => review(item.id, 'rejected')}>Reject</Button></Flex></td>
          </tr>)}
        </tbody></table>
      </Dialog.Content>
    </Dialog>
  );
}
