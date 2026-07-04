import { FormEvent, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { agenticChatAPI } from '../../services/AgenticChatAPI';
import { cancelAgenticRAGJob } from '../../services/MEKGAPI';
import { AgenticChatDetails } from '../../types';
import './AgenticChatPage.css';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  details?: AgenticChatDetails;
  loading?: boolean;
  error?: string;
};

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
    source.source_type ? source.source_type : '',
  ]
    .filter(Boolean)
    .join(' · ');
  return { title, coordinates };
};

const StatusPill = ({ details }: { details?: AgenticChatDetails }) => {
  if (!details) return null;
  const mode = details.mode ? modeLabel[details.mode] ?? details.mode : details.status ?? 'running';
  const confidence = typeof details.confidence === 'number' ? ` · confidence ${details.confidence.toFixed(2)}` : '';
  const webCalls =
    typeof details.diagnostics?.web_calls === 'number' && details.diagnostics.web_calls > 0
      ? ` · web ${details.diagnostics.web_calls}`
      : '';
  return (
    <div className='mb-3 inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700'>
      {mode}
      {confidence}
      {webCalls}
    </div>
  );
};

const EvidenceBlock = ({ details }: { details?: AgenticChatDetails }) => {
  const sources = details?.sources ?? [];
  const warnings = details?.warnings ?? [];
  const gaps = details?.gaps ?? [];
  const trace = details?.trace ?? [];
  if (!details || (!sources.length && !warnings.length && !gaps.length && !trace.length)) return null;

  return (
    <div className='mt-4 space-y-3 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700'>
      {sources.length > 0 && (
        <details open>
          <summary className='cursor-pointer font-semibold'>Источники ({sources.length})</summary>
          <div className='mt-3 grid gap-2'>
            {sources.slice(0, 8).map((source, index) => {
              const formatted = formatSource(source);
              return (
                <div key={`${source.label ?? index}-${source.source_id ?? source.url ?? index}`} className='rounded-xl bg-white p-3'>
                  <div className='font-medium text-slate-900'>
                    {source.label ? `[${source.label}] ` : ''}
                    {source.url ? (
                      <a className='text-blue-600 underline' href={source.url} target='_blank' rel='noreferrer'>
                        {formatted.title}
                      </a>
                    ) : (
                      formatted.title
                    )}
                  </div>
                  {formatted.coordinates && <div className='mt-1 text-xs text-slate-500'>{formatted.coordinates}</div>}
                </div>
              );
            })}
          </div>
        </details>
      )}
      {gaps.length > 0 && (
        <div>
          <div className='font-semibold'>Пробелы</div>
          <div>{gaps.slice(0, 6).join(', ')}</div>
        </div>
      )}
      {warnings.length > 0 && (
        <details>
          <summary className='cursor-pointer font-semibold'>Warnings ({warnings.length})</summary>
          <ul className='mt-2 list-disc pl-5'>
            {warnings.slice(0, 8).map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </details>
      )}
      {trace.length > 0 && (
        <details>
          <summary className='cursor-pointer font-semibold'>Trace ({trace.length})</summary>
          <ul className='mt-2 list-disc pl-5 text-xs'>
            {trace.slice(-10).map((event) => (
              <li key={event.id}>
                {event.iteration ? `Итерация ${event.iteration}: ` : ''}
                {event.type}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
};

const AgenticChatPage = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'hello',
      role: 'assistant',
      content:
        'Привет! Я исследовательский чат по “Научному клубку”. Задайте технический вопрос — я найду evidence в локальных документах, покажу источники и уверенность ответа.',
    },
  ]);
  const [input, setInput] = useState('');
  const [allowWeb, setAllowWeb] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const loading = useMemo(() => messages.some((message) => message.loading), [messages]);

  const ask = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    setInput('');
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: 'Собираю evidence…',
      loading: true,
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    try {
      const response = await agenticChatAPI(trimmed, {
        allowExternalWeb: allowWeb,
        webProfileIds: allowWeb ? ['journals', 'mining_metals'] : [],
        corpora: [],
        onUpdate: (update) => {
          setActiveJobId(update.agentic?.job_id ?? null);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: update.message,
                    details: update.agentic,
                    loading: !['complete', 'failed', 'cancelled'].includes(update.agentic?.status ?? ''),
                    error: update.error,
                  }
                : message
            )
          );
        },
      });
      const info = response.response.data.data.info;
      setActiveJobId(null);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: response.response.data.data.message,
                details: info.agentic,
                loading: false,
                error: info.error,
              }
            : message
        )
      );
    } catch (error) {
      setActiveJobId(null);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: 'Не получилось выполнить запрос. Проверьте backend/Yandex ключ и попробуйте ещё раз.',
                loading: false,
                error: error instanceof Error ? error.message : String(error),
              }
            : message
        )
      );
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void ask(input);
  };

  const stop = async () => {
    if (!activeJobId) return;
    await cancelAgenticRAGJob(activeJobId);
    setActiveJobId(null);
  };

  return (
    <div className='agentic-chat-page'>
      <header className='agentic-chat-header'>
        <div className='agentic-chat-header-inner'>
          <div className='agentic-chat-brand'>
            <div className='agentic-chat-title'>Nornikel Agentic RAG</div>
            <div className='agentic-chat-subtitle'>Knowledge graph + vector/BM25 search по рабочим документам</div>
          </div>
          <label className='agentic-chat-web-toggle'>
            <input
              type='checkbox'
              checked={allowWeb}
              onChange={(event) => setAllowWeb(event.target.checked)}
              aria-label='Разрешить Web Search'
            />
            Web Search
          </label>
        </div>
      </header>

      <main className='agentic-chat-main'>
        <div className='agentic-chat-messages'>
          {messages.map((message) => (
            <div key={message.id} className={`agentic-chat-row agentic-chat-row-${message.role}`}>
              <article
                className={`agentic-chat-message agentic-chat-message-${message.role}`}
              >
                {message.role === 'assistant' && <StatusPill details={message.details} />}
                <div className='agentic-chat-markdown'>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                </div>
                {message.error && <div className='agentic-chat-error'>{message.error}</div>}
                {message.loading && <div className='agentic-chat-working'>Работаю…</div>}
                {message.role === 'assistant' && <EvidenceBlock details={message.details} />}
              </article>
            </div>
          ))}
        </div>

        <div className='agentic-chat-composer-wrap'>
          <form onSubmit={submit} className='agentic-chat-composer'>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void ask(input);
                }
              }}
              placeholder='Спросите по рабочим документам…'
              rows={1}
              className='agentic-chat-input'
              disabled={loading}
            />
            {activeJobId && (
              <button type='button' className='agentic-chat-stop' onClick={() => void stop()}>
                Stop
              </button>
            )}
            <button
              type='submit'
              className='agentic-chat-send'
              disabled={loading || !input.trim()}
            >
              Send
            </button>
          </form>
          <div className='agentic-chat-footnote'>
            Web Search выключен по умолчанию. При включении наружу уходят только очищенные поисковые формулировки.
          </div>
        </div>
      </main>
    </div>
  );
};

export default AgenticChatPage;
