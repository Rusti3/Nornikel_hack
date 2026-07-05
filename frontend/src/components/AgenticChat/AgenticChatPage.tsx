import { DragEvent, FormEvent, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { agenticChatAPI } from '../../services/AgenticChatAPI';
import {
  cancelAgenticRAGJob,
  cancelIngestJob,
  getAgenticReportUrl,
  startIngestJob,
} from '../../services/MEKGAPI';
import { IngestJob, ingestProgressText, waitForIngestJob } from '../../services/IngestAPI';
import { AgenticChatDetails } from '../../types';
import DemoHeader from '../DemoShell/DemoHeader';
import './AgenticChatPage.css';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  details?: AgenticChatDetails;
  loading?: boolean;
  error?: string;
};

const ACCEPTED = '.pdf,.doc,.docx,.docm,.pptx,.xls,.xlsx';
const MAX_SIZE = 100 * 1024 * 1024;

const modeLabel: Record<string, string> = {
  full_answer: 'Полный ответ',
  partial_answer_with_gaps: 'Частичный ответ',
  NO_DIRECT_DATA: 'Нет прямых данных',
  NO_NUMERIC_DATA: 'Нет числовых данных',
  NO_EVIDENCE_FOUND: 'Evidence не найдено',
  OUT_OF_SCOPE: 'Вне области',
};

const formatSource = (source: NonNullable<AgenticChatDetails['sources']>[number]) => {
  const title = source.title || source.file_name || source.url || source.source_id || 'Источник';
  const coordinates = [
    source.page ? `стр. ${source.page}` : '',
    source.slide ? `слайд ${source.slide}` : '',
    source.source_type || '',
  ].filter(Boolean).join(' · ');
  return { title, coordinates };
};

const StatusPill = ({ details }: { details?: AgenticChatDetails }) => {
  if (!details) return null;
  const mode = details.mode ? modeLabel[details.mode] ?? details.mode : details.status ?? 'running';
  const confidence = typeof details.confidence === 'number' ? ` · confidence ${details.confidence.toFixed(2)}` : '';
  const webCalls = typeof details.diagnostics?.web_calls === 'number' && details.diagnostics.web_calls > 0
    ? ` · web ${details.diagnostics.web_calls}` : '';
  return <div className='agentic-status-pill'>{mode}{confidence}{webCalls}</div>;
};

const AnswerActions = ({ details }: { details?: AgenticChatDetails }) => {
  if (!details?.job_id || details.status !== 'complete') return null;
  return (
    <div className='agentic-answer-actions'>
      <a href={`/graph?agent_job_id=${encodeURIComponent(details.job_id)}`}>Показать в графе</a>
      <a href={getAgenticReportUrl(details.job_id, 'markdown')}>Скачать Markdown</a>
      <a href={getAgenticReportUrl(details.job_id, 'pdf')}>Скачать PDF</a>
    </div>
  );
};

const EvidenceBlock = ({ details }: { details?: AgenticChatDetails }) => {
  const sources = details?.sources ?? [];
  const warnings = details?.warnings ?? [];
  const gaps = details?.gaps ?? [];
  const trace = details?.trace ?? [];
  const preview = details?.preview ?? [];
  if (!details || (!sources.length && !warnings.length && !gaps.length && !trace.length && !preview.length)) return null;
  return (
    <div className='agentic-evidence-block'>
      {!sources.length && preview.length > 0 && (
        <details open>
          <summary>Быстрый retrieval-preview ({preview.length})</summary>
          <div className='agentic-source-grid'>
            {preview.slice(0, 6).map((item, index) => (
              <div className='agentic-source-card' key={`${item.document_id}-${index}`}>
                <strong>{item.file_name || item.document_id || 'Источник'}</strong>
                <span>{item.page ? `стр. ${item.page}` : item.slide ? `слайд ${item.slide}` : 'координаты уточняются'}</span>
              </div>
            ))}
          </div>
        </details>
      )}
      {sources.length > 0 && (
        <details open>
          <summary>Источники ({sources.length})</summary>
          <div className='agentic-source-grid'>
            {sources.slice(0, 10).map((source, index) => {
              const formatted = formatSource(source);
              return (
                <div key={`${source.label ?? index}-${source.source_id ?? source.url ?? index}`} className='agentic-source-card'>
                  <strong>
                    {source.label ? `[${source.label}] ` : ''}
                    {source.url ? <a href={source.url} target='_blank' rel='noreferrer'>{formatted.title}</a> : formatted.title}
                  </strong>
                  {formatted.coordinates && <span>{formatted.coordinates}</span>}
                </div>
              );
            })}
          </div>
        </details>
      )}
      {gaps.length > 0 && <div><strong>Пробелы:</strong> {gaps.slice(0, 8).join(', ')}</div>}
      {warnings.length > 0 && <details><summary>Ограничения ({warnings.length})</summary><ul>{warnings.slice(0, 8).map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul></details>}
      {trace.length > 0 && <details><summary>Ход исследования ({trace.length})</summary><ul>{trace.slice(-12).map((event) => <li key={event.id}>{event.iteration ? `Итерация ${event.iteration}: ` : ''}{event.type}</li>)}</ul></details>}
    </div>
  );
};

export default function AgenticChatPage() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: 'hello', role: 'assistant', content: 'Привет! Задайте технический вопрос или приложите рабочий документ. Я найду evidence, покажу источники и честно отмечу пробелы.' }]);
  const [input, setInput] = useState('');
  const [allowWeb, setAllowWeb] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeKind, setActiveKind] = useState<'agent' | 'ingest' | null>(null);
  const loading = useMemo(() => messages.some((message) => message.loading), [messages]);

  const addFiles = (values: File[]) => {
    const valid = values.filter((file) => file.size > 0 && file.size <= MAX_SIZE && ACCEPTED.split(',').some((suffix) => file.name.toLowerCase().endsWith(suffix)));
    setAttachments((current) => Array.from(new Map([...current, ...valid].map((file) => [`${file.name}:${file.size}`, file])).values()).slice(0, 10));
  };

  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  };

  const updateAssistant = (assistantId: string, change: Partial<ChatMessage>) => {
    setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, ...change } : message));
  };

  const ask = async (question: string) => {
    const trimmed = question.trim();
    const selectedFiles = attachments;
    if ((!trimmed && !selectedFiles.length) || loading) return;
    setInput('');
    setAttachments([]);
    const attachmentLines = selectedFiles.map((file) => `📎 ${file.name}`).join('\n');
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(), role: 'user',
      content: [trimmed || 'Добавить документы в базу знаний', attachmentLines].filter(Boolean).join('\n\n'),
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: selectedFiles.length ? 'Загружаю документы…' : 'Собираю evidence…', loading: true };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    try {
      let focusDocumentIds: string[] = [];
      if (selectedFiles.length) {
        const accepted = await startIngestJob(selectedFiles, { category: 'auto' });
        const ingestId = accepted.data.job_id as string;
        setActiveJobId(ingestId);
        setActiveKind('ingest');
        const finished = await waitForIngestJob(ingestId, (job: IngestJob) => {
          updateAssistant(assistantId, { content: ingestProgressText(job), loading: true, error: job.error });
        });
        if (finished.status !== 'complete') throw new Error(finished.error || 'Индексация не завершена');
        focusDocumentIds = finished.result?.document_ids ?? [];
        if (!trimmed) {
          updateAssistant(assistantId, {
            content: `### Документы добавлены\n\nОбработано: ${focusDocumentIds.length}. Данные доступны в BM25, vector search и Neo4j.`,
            loading: false,
          });
          setActiveJobId(null); setActiveKind(null);
          return;
        }
      }
      const response = await agenticChatAPI(trimmed, {
        allowExternalWeb: allowWeb,
        webProfileIds: allowWeb ? ['journals', 'mining_metals'] : [],
        corpora: [],
        focusDocumentIds,
        onUpdate: (update) => {
          setActiveJobId(update.agentic?.job_id ?? null);
          setActiveKind('agent');
          updateAssistant(assistantId, {
            content: update.message,
            details: update.agentic,
            loading: !['complete', 'failed', 'cancelled'].includes(update.agentic?.status ?? ''),
            error: update.error,
          });
        },
      });
      const info = response.response.data.data.info;
      updateAssistant(assistantId, { content: response.response.data.data.message, details: info.agentic, loading: false, error: info.error });
      setActiveJobId(null); setActiveKind(null);
    } catch (error) {
      updateAssistant(assistantId, { content: 'Не удалось завершить запрос. Проверьте состояние сервисов и попробуйте ещё раз.', loading: false, error: error instanceof Error ? error.message : String(error) });
      setActiveJobId(null); setActiveKind(null);
    }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void ask(input); };
  const stop = async () => {
    if (!activeJobId) return;
    if (activeKind === 'ingest') await cancelIngestJob(activeJobId);
    else await cancelAgenticRAGJob(activeJobId);
    setActiveJobId(null); setActiveKind(null);
  };

  return (
    <div className={`agentic-chat-page ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
      <DemoHeader right={<label className='agentic-chat-web-toggle'><input type='checkbox' checked={allowWeb} onChange={(event) => setAllowWeb(event.target.checked)} aria-label='Разрешить Web Search' />Web Search</label>} />
      {dragging && <div className='agentic-drop-overlay'>Отпустите файлы, чтобы добавить их в сообщение</div>}
      <main className='agentic-chat-main'>
        <div className='agentic-chat-messages'>
          {messages.map((message) => (
            <div key={message.id} className={`agentic-chat-row agentic-chat-row-${message.role}`}>
              <article className={`agentic-chat-message agentic-chat-message-${message.role}`}>
                {message.role === 'assistant' && <div className='agentic-message-head'><StatusPill details={message.details} /><AnswerActions details={message.details} /></div>}
                <div className='agentic-chat-markdown'><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>
                {message.error && <div className='agentic-chat-error'>{message.error}</div>}
                {message.loading && <div className='agentic-chat-working'>Работаю…</div>}
                {message.role === 'assistant' && <EvidenceBlock details={message.details} />}
              </article>
            </div>
          ))}
        </div>

        <div className='agentic-chat-composer-wrap'>
          {attachments.length > 0 && <div className='agentic-attachments'>{attachments.map((file) => <div key={`${file.name}:${file.size}`}>📎 <span>{file.name}</span><button onClick={() => setAttachments((current) => current.filter((item) => item !== file))}>×</button></div>)}</div>}
          <form onSubmit={submit} className='agentic-chat-composer'>
            <input ref={fileInput} type='file' hidden multiple accept={ACCEPTED} onChange={(event) => addFiles(Array.from(event.target.files ?? []))} />
            <button type='button' className='agentic-attach' title='Добавить документы' onClick={() => fileInput.current?.click()}>＋</button>
            <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void ask(input); } }} placeholder='Спросите по рабочим документам…' rows={1} className='agentic-chat-input' disabled={loading} />
            {activeJobId && <button type='button' className='agentic-chat-stop' onClick={() => void stop()}>Stop</button>}
            <button type='submit' className='agentic-chat-send' disabled={loading || (!input.trim() && !attachments.length)}>Отправить</button>
          </form>
          <div className='agentic-chat-footnote'>Web Search выключен по умолчанию. Вложения навсегда добавляются в общий граф и vector search.</div>
        </div>
      </main>
    </div>
  );
}
