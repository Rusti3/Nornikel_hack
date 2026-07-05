import axios from 'axios';
import { url } from '../utils/Utils';

const mekgApi = axios.create({
  baseURL: `${url()}/api/mekg/v1`,
  headers: { 'Content-Type': 'application/json' },
});

export const getMEKGProfile = () => mekgApi.get('/profile');
export const initializeMEKG = () => mekgApi.post('/initialize');
export const getMEKGQA = () => mekgApi.get('/qa');
export const runMEKGQA = () => mekgApi.post('/qa/run');
export const getMEKGReviewQueue = () => mekgApi.get('/review-queue');
export const crossCorpusSearch = (request: {
  query: string;
  intent?: string;
  target_slots?: string[];
  corpora?: string[];
  final_k?: number;
  include_debug?: boolean;
}) => mekgApi.post('/cross_corpus_search', request);
export const getMEKGPipelineStatus = (runId?: string) =>
  mekgApi.get('/pipeline/status', { params: runId ? { run_id: runId } : {} });
export const getMEKGReadiness = (runId?: string) =>
  mekgApi.get('/readiness', { params: runId ? { run_id: runId } : {} });
export const getMEKGGraphNeighbors = (nodeId: string) =>
  mekgApi.get(`/graph/neighbors/${encodeURIComponent(nodeId)}`);
export const getWebSearchProfiles = () => mekgApi.get('/web_search/profiles');
export const startAgenticRAG = (request: {
  query: string;
  allow_external_web: boolean;
  web_profile_ids?: string[];
  corpora?: string[];
  filters?: {
    source_type?: string[];
    language?: string;
    year_min?: number;
    year_max?: number;
    geography?: string;
    domain?: string;
    confidence_min?: number;
  };
  max_iterations?: number;
  include_debug?: boolean;
  focus_document_ids?: string[];
}) => mekgApi.post('/agentic_rag/jobs', request);
export const getAgenticRAGJob = (jobId: string) =>
  mekgApi.get(`/agentic_rag/jobs/${encodeURIComponent(jobId)}`);
export const cancelAgenticRAGJob = (jobId: string) =>
  mekgApi.post(`/agentic_rag/jobs/${encodeURIComponent(jobId)}/cancel`);
export const getAgenticRAGEventsUrl = (jobId: string) =>
  `${url()}/api/mekg/v1/agentic_rag/jobs/${encodeURIComponent(jobId)}/events`;
export const getAgenticReportUrl = (jobId: string, format: 'markdown' | 'pdf') =>
  `${url()}/api/mekg/v1/agentic_rag/jobs/${encodeURIComponent(jobId)}/export?format=${format}`;
export const startIngestJob = (files: File[], options?: {
  category?: string;
  question?: string;
  allowExternalWeb?: boolean;
}) => {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  form.append('category', options?.category ?? 'auto');
  if (options?.question?.trim()) form.append('question', options.question.trim());
  form.append('allow_external_web', String(Boolean(options?.allowExternalWeb)));
  return mekgApi.post('/ingest/jobs', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
};
export const listIngestJobs = (limit = 50) => mekgApi.get('/ingest/jobs', { params: { limit } });
export const getIngestJob = (jobId: string) => mekgApi.get(`/ingest/jobs/${encodeURIComponent(jobId)}`);
export const cancelIngestJob = (jobId: string) => mekgApi.post(`/ingest/jobs/${encodeURIComponent(jobId)}/cancel`);
export const getIngestEventsUrl = (jobId: string) =>
  `${url()}/api/mekg/v1/ingest/jobs/${encodeURIComponent(jobId)}/events`;
export const getGraphPathways = (request: {
  query?: string;
  agent_job_id?: string;
  document_ids?: string[];
  entity_ids?: string[];
  limit?: number;
  include_incomplete?: boolean;
}) => mekgApi.post('/graph/pathways', request);
export const reviewMEKGFact = (id: string, status: string, reviewer = 'local-reviewer') =>
  mekgApi.post(`/review/${encodeURIComponent(id)}`, { status, reviewer });
export const getMEKGExportUrl = (format: 'turtle' | 'jsonld') =>
  `${url()}/api/mekg/v1/export/${format}`;

export default mekgApi;
