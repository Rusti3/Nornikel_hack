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
}) => mekgApi.post('/agentic_rag/jobs', request);
export const getAgenticRAGJob = (jobId: string) =>
  mekgApi.get(`/agentic_rag/jobs/${encodeURIComponent(jobId)}`);
export const cancelAgenticRAGJob = (jobId: string) =>
  mekgApi.post(`/agentic_rag/jobs/${encodeURIComponent(jobId)}/cancel`);
export const getAgenticRAGEventsUrl = (jobId: string) =>
  `${url()}/api/mekg/v1/agentic_rag/jobs/${encodeURIComponent(jobId)}/events`;
export const reviewMEKGFact = (id: string, status: string, reviewer = 'local-reviewer') =>
  mekgApi.post(`/review/${encodeURIComponent(id)}`, { status, reviewer });
export const getMEKGExportUrl = (format: 'turtle' | 'jsonld') =>
  `${url()}/api/mekg/v1/export/${format}`;

export default mekgApi;
