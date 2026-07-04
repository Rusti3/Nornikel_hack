import React, { FC, lazy, Suspense, useCallback, useEffect, useReducer, useRef, useState } from 'react';
import {
  Widget,
  Typography,
  Avatar,
  TextInput,
  IconButton,
  Modal,
  useCopyToClipboard,
  Flex,
  Box,
  TextLink,
  SpotlightTarget,
} from '@neo4j-ndl/react';
import { ArrowDownTrayIconOutline, XMarkIconOutline } from '@neo4j-ndl/react/icons';
import ChatBotAvatar from '../../assets/images/chatbot-ai.png';
import {
  ChatbotProps,
  Chunk,
  Community,
  CustomFile,
  Entity,
  ExtendedNode,
  ExtendedRelationship,
  Messages,
  AgenticChatDetails,
  ResponseMode,
  metricstate,
  multimodelmetric,
  nodeDetailsProps,
} from '../../types';
import { chatBotAPI } from '../../services/QnaAPI';
import { agenticChatAPI } from '../../services/AgenticChatAPI';
import { cancelAgenticRAGJob, getMEKGGraphNeighbors, getWebSearchProfiles } from '../../services/MEKGAPI';
import { checkTokenLimits } from '../../utils/TokenWarning';
import { showErrorToast, showNormalToast } from '../../utils/Toasts';
import { v4 as uuidv4 } from 'uuid';
import { useFileContext } from '../../context/UsersFiles';
import { useCredentials } from '../../context/UserCredentials';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import { buttonCaptions, chatModeLables } from '../../utils/Constants';
import useSpeechSynthesis from '../../hooks/useSpeech';
import ButtonWithToolTip from '../UI/ButtonWithToolTip';
import FallBackDialog from '../UI/FallBackDialog';
import { downloadClickHandler, getDateTime, shouldShowTokenTracking } from '../../utils/Utils';
import ChatModesSwitch from './ChatModesSwitch';
import CommonActions from './CommonChatActions';
import Loader from '../../utils/Loader';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import GraphViewModal from '../Graph/GraphViewModal';
const InfoModal = lazy(() => import('./ChatInfoModal'));
if (typeof window !== 'undefined') {
  if (!sessionStorage.getItem('session_id')) {
    const id = uuidv4();
    sessionStorage.setItem('session_id', id);
  }
}
const sessionId = sessionStorage.getItem('session_id') ?? '';

const agenticCorpusOptions = [
  { id: 'internal_reports', label: 'Внутренние отчёты' },
  { id: 'scientific_journals', label: 'Журналы' },
  { id: 'conference_materials', label: 'Конференции' },
  { id: 'reviews', label: 'Обзоры' },
  { id: 'scientific_articles', label: 'Статьи' },
];

const agenticExamples = [
  'Какие методы обессоливания воды подходят для обогатительной фабрики при сульфатах, хлоридах, Ca, Mg, Na 200–300 мг/л и сухом остатке ≤1000 мг/дм³?',
  'Какие технические решения циркуляции католита при электроэкстракции никеля описаны в мировой практике, и какая скорость потока считается оптимальной?',
  'Покажи эксперименты и публикации по распределению Au, Ag и МПГ между медным/никелевым штейном и шлаком за последние 5 лет.',
  'Какие способы закачки шахтных вод в глубокие горизонты применялись в России и за рубежом, и каковы их технико-экономические показатели?',
];

const agenticModeLabel: Record<string, string> = {
  full_answer: 'Полный ответ',
  partial_answer_with_gaps: 'Частичный ответ с пробелами',
  NO_DIRECT_DATA: 'Нет прямых данных',
  NO_NUMERIC_DATA: 'Нет числовых данных',
  NO_EVIDENCE_FOUND: 'Evidence не найдено',
  OUT_OF_SCOPE: 'Вне области',
};

const sourceTitle = (source: NonNullable<AgenticChatDetails['sources']>[number]) =>
  source.title || source.file_name || source.url || source.source_id || 'Источник';

const AgenticDetailsBlock = ({ details, answerMarkdown = '' }: { details?: AgenticChatDetails; answerMarkdown?: string }) => {
  const [graphNodes, setGraphNodes] = useState<any[]>([]);
  const [graphRelationships, setGraphRelationships] = useState<any[]>([]);
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  if (!details) return null;
  const sources = details.sources ?? [];
  const gaps = details.gaps ?? [];
  const contradictions = details.contradictions ?? [];
  const warnings = details.warnings ?? [];
  const trace = details.trace ?? [];
  const graphCandidates = details.graph_candidates ?? [];
  const downloadMarkdown = () => {
    const href = URL.createObjectURL(new Blob([answerMarkdown], { type: 'text/markdown;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = href;
    link.download = `mekg-agentic-answer-${details.job_id || 'result'}.md`;
    link.click();
    URL.revokeObjectURL(href);
  };
  const showGraphCandidates = async () => {
    setGraphLoading(true);
    try {
      const responses = await Promise.all(graphCandidates.slice(0, 10).map((id) => getMEKGGraphNeighbors(id)));
      const nodes = responses.flatMap((response) => response.data?.nodes ?? []);
      const relationships = responses.flatMap((response) => response.data?.relationships ?? []);
      setGraphNodes(Array.from(new Map(nodes.map((node: any) => [node.element_id, node])).values()));
      setGraphRelationships(
        Array.from(new Map(relationships.map((relationship: any) => [relationship.element_id, relationship])).values())
      );
      if (nodes.length) setGraphOpen(true);
      else showNormalToast('Для выбранных evidence-кандидатов соседние узлы не найдены.');
    } catch {
      showErrorToast('Не удалось загрузить graph evidence.');
    } finally {
      setGraphLoading(false);
    }
  };
  return (
    <>
    <div className='mt-3 rounded border border-neutral-300 bg-white/80 p-2 text-xs' data-testid='agentic-evidence-card'>
      <div className='flex! flex-wrap gap-2 items-center mb-2'>
        <span className='font-bold'>{agenticModeLabel[details.mode ?? ''] ?? details.mode ?? 'Agentic RAG'}</span>
        {typeof details.confidence === 'number' && <span>confidence {details.confidence.toFixed(2)}</span>}
        {typeof details.diagnostics?.searched_iterations === 'number' && (
          <span>{details.diagnostics.searched_iterations} итерац.</span>
        )}
        {typeof details.diagnostics?.web_calls === 'number' && <span>web calls {details.diagnostics.web_calls}</span>}
        {answerMarkdown && details.status === 'complete' && (
          <button className='rounded border px-2 py-1' type='button' onClick={downloadMarkdown}>
            Скачать Markdown
          </button>
        )}
      </div>
      {sources.length > 0 && (
        <details open>
          <summary className='cursor-pointer font-bold'>Evidence sources ({sources.length})</summary>
          <div className='mt-2 grid gap-2'>
            {sources.slice(0, 8).map((source, index) => (
              <div key={`${source.label ?? index}-${source.source_id ?? source.url ?? index}`} className='rounded bg-neutral-50 p-2'>
                <div className='font-semibold'>
                  {source.label ? `[${source.label}] ` : ''}
                  {source.url ? (
                    <a href={source.url} target='_blank' rel='noreferrer'>
                      {sourceTitle(source)}
                    </a>
                  ) : (
                    sourceTitle(source)
                  )}
                </div>
                <div className='opacity-80'>
                  {source.source_type ?? 'local'} · {source.file_name ?? source.source_id ?? '—'}
                  {source.page ? ` · page ${source.page}` : ''}
                  {source.slide ? ` · slide ${source.slide}` : ''}
                  {source.metadata_only ? ' · metadata-only' : ''}
                  {source.direct === false ? ' · analogy' : ''}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
      {(gaps.length > 0 || contradictions.length > 0 || warnings.length > 0) && (
        <div className='mt-2 grid gap-2'>
          {gaps.length > 0 && (
            <div>
              <span className='font-bold'>Пробелы: </span>
              {gaps.join(', ')}
            </div>
          )}
          {contradictions.length > 0 && (
            <div>
              <span className='font-bold'>Противоречия: </span>
              {contradictions.slice(0, 3).map((item) => item.topic || item.id || JSON.stringify(item)).join('; ')}
            </div>
          )}
          {warnings.length > 0 && (
            <div>
              <span className='font-bold'>Warnings: </span>
              {warnings.slice(0, 5).join('; ')}
            </div>
          )}
        </div>
      )}
      {graphCandidates.length > 0 && (
        <div className='mt-2'>
          <button
            className='rounded border px-2 py-1'
            type='button'
            title={graphCandidates.join(', ')}
            disabled={graphLoading}
            onClick={showGraphCandidates}
          >
            {graphLoading ? 'Загрузка графа…' : `Показать кандидатов в графе (${graphCandidates.length})`}
          </button>
        </div>
      )}
      {trace.length > 0 && (
        <details className='mt-2'>
          <summary className='cursor-pointer font-bold'>Live trace ({trace.length})</summary>
          <div className='mt-1 max-h-40 overflow-y-auto'>
            {trace.slice(-20).map((event) => (
              <div key={event.id}>
                {event.iteration ? `Итерация ${event.iteration}: ` : ''}
                {event.type}
                {event.payload?.missing_slots?.length ? ` · missing: ${event.payload.missing_slots.join(', ')}` : ''}
                {event.payload?.focus?.length ? ` · focus: ${event.payload.focus.join(', ')}` : ''}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
    {graphOpen && (
      <GraphViewModal
        open={graphOpen}
        setGraphViewOpen={setGraphOpen}
        viewPoint='chatInfoView'
        nodeValues={graphNodes}
        relationshipValues={graphRelationships}
      />
    )}
    </>
  );
};

const Chatbot: FC<ChatbotProps> = (props) => {
  const {
    messages: listMessages,
    setMessages: setListMessages,
    isLoading,
    isFullScreen,
    connectionStatus,
    isChatOnly,
    isDeleteChatLoading,
  } = props;
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState<boolean>(isLoading);
  const { model, chatModes, selectedRows, filesData } = useFileContext();
  const { userCredentials } = useCredentials();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
  const [sourcesModal, setSourcesModal] = useState<string[]>([]);
  const [modelModal, setModelModal] = useState<string>('');
  const [responseTime, setResponseTime] = useState<number>(0);
  const [tokensUsed, setTokensUsed] = useState<number>(0);
  const [cypherQuery, setcypherQuery] = useState<string>('');
  const [chatsMode, setChatsMode] = useState<string>(chatModeLables['graph+vector+fulltext']);
  const [graphEntitites, setgraphEntitites] = useState<[]>([]);
  const [messageError, setmessageError] = useState<string>('');
  const [entitiesModal, setEntitiesModal] = useState<string[]>([]);
  const [nodeDetailsModal, setNodeDetailsModal] = useState<nodeDetailsProps>({});
  const [metricQuestion, setMetricQuestion] = useState<string>('');
  const [metricAnswer, setMetricAnswer] = useState<string>('');
  const [metricContext, setMetricContext] = useState<string>('');
  const [nodes, setNodes] = useState<ExtendedNode[]>([]);
  const [relationships, setRelationships] = useState<ExtendedRelationship[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [metricDetails, setMetricDetails] = useState<metricstate | null>(null);
  const [infoEntities, setInfoEntities] = useState<Entity[]>([]);
  const [communities, setCommunities] = useState<Community[]>([]);
  const [infoLoading, toggleInfoLoading] = useReducer((s) => !s, false);
  const [metricsLoading, toggleMetricsLoading] = useReducer((s) => !s, false);
  const downloadLinkRef = useRef<HTMLAnchorElement>(null);
  const [activeChat, setActiveChat] = useState<Messages | null>(null);
  const [multiModelMetrics, setMultiModelMetrics] = useState<multimodelmetric[]>([]);
  const [agenticAllowExternalWeb, setAgenticAllowExternalWeb] = useState(false);
  const [agenticWebProfiles, setAgenticWebProfiles] = useState<any[]>([]);
  const [agenticSelectedWebProfiles, setAgenticSelectedWebProfiles] = useState<string[]>(['journals', 'mining_metals']);
  const [agenticSelectedCorpora, setAgenticSelectedCorpora] = useState<string[]>([]);
  const [agenticGeography, setAgenticGeography] = useState('');
  const [agenticYearMin, setAgenticYearMin] = useState('');
  const [agenticYearMax, setAgenticYearMax] = useState('');
  const [agenticNumericMode, setAgenticNumericMode] = useState<'boost' | 'strict'>('boost');
  const [activeAgenticJobId, setActiveAgenticJobId] = useState<string | null>(null);

  const [_, copy] = useCopyToClipboard();
  const { speak, cancel, speaking } = useSpeechSynthesis({
    onEnd: () => {
      setListMessages((msgs) => msgs.map((msg) => ({ ...msg, speaking: false })));
    },
  });

  let selectedFileNames: CustomFile[] = filesData.filter(
    (f) => selectedRows.includes(f.id) && ['Completed'].includes(f.status)
  );
  const isAgenticModeSelected = chatModes.includes(chatModeLables['agentic rag']);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputMessage(e.target.value);
  };

  useEffect(() => {
    if (!isAgenticModeSelected || agenticWebProfiles.length > 0) return;
    getWebSearchProfiles()
      .then((response) => setAgenticWebProfiles(response.data.profiles ?? []))
      .catch(() => {
        // The chat still works without web profiles because external web is opt-in.
      });
  }, [agenticWebProfiles.length, isAgenticModeSelected]);

  const toggleAgenticWebProfile = (id: string) => {
    setAgenticSelectedWebProfiles((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      if (current.length >= 2) return [current[1], id];
      return [...current, id];
    });
  };

  const toggleAgenticCorpus = (id: string) => {
    setAgenticSelectedCorpora((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    );
  };

  const cancelActiveAgenticJob = async () => {
    if (!activeAgenticJobId) return;
    try {
      await cancelAgenticRAGJob(activeAgenticJobId);
      setActiveAgenticJobId(null);
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : 'Unable to cancel Agentic RAG job');
    }
  };

  const saveInfoEntitites = (entities: Entity[]) => {
    setInfoEntities(entities);
  };

  const saveNodes = (chatNodes: ExtendedNode[]) => {
    setNodes(chatNodes);
  };

  const saveChatRelationships = (chatRels: ExtendedRelationship[]) => {
    setRelationships(chatRels);
  };

  const saveChunks = (chatChunks: Chunk[]) => {
    setChunks(chatChunks);
  };
  const saveMultimodemetrics = (metrics: multimodelmetric[]) => {
    setMultiModelMetrics(metrics);
  };
  const saveMetrics = (metricInfo: metricstate) => {
    setMetricDetails(metricInfo);
  };
  const saveCommunities = (chatCommunities: Community[]) => {
    setCommunities(chatCommunities);
  };

  const simulateTypingEffect = (messageId: number, response: ResponseMode, mode: string, message: string) => {
    let index = 0;
    let lastTimestamp: number | null = null;
    const TYPING_INTERVAL = 20;
    const animate = (timestamp: number) => {
      if (lastTimestamp === null) {
        lastTimestamp = timestamp;
      }
      const elapsed = timestamp - lastTimestamp;
      if (elapsed >= TYPING_INTERVAL) {
        if (index < message.length) {
          const nextIndex = index + 1;
          const currentTypedText = message.substring(0, nextIndex);
          setListMessages((msgs) =>
            msgs.map((msg) => {
              if (msg.id === messageId) {
                return {
                  ...msg,
                  modes: {
                    ...msg.modes,
                    [mode]: {
                      ...response,
                      message: currentTypedText,
                    },
                  },
                  isTyping: true,
                  speaking: false,
                  copying: false,
                };
              }
              return msg;
            })
          );
          index = nextIndex;
          lastTimestamp = timestamp;
        } else {
          setListMessages((msgs) => {
            const activeMessage = msgs.find((message) => message.id === messageId);
            let sortedModes: Record<string, ResponseMode>;
            if (activeMessage) {
              sortedModes = Object.fromEntries(
                chatModes.filter((m) => m in activeMessage.modes).map((key) => [key, activeMessage?.modes[key]])
              );
            }
            return msgs.map((msg) => (msg.id === messageId ? { ...msg, isTyping: false, modes: sortedModes } : msg));
          });
          return;
        }
      }
      requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  };

  const handleSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    if (!inputMessage.trim()) {
      return;
    }

    const SHOW_TOKEN_TRACKING_IN_CHAT = false;
    if (
      SHOW_TOKEN_TRACKING_IN_CHAT &&
      userCredentials &&
      connectionStatus &&
      shouldShowTokenTracking(userCredentials.email)
    ) {
      const tokenCheck = await checkTokenLimits(userCredentials);
      if (tokenCheck.shouldWarn) {
        showNormalToast(tokenCheck.message);
      }
    }

    const datetime = getDateTime();
    const userMessage: Messages = {
      id: Date.now(),
      user: 'user',
      datetime: datetime,
      currentMode: chatModes[0],
      modes: {},
    };
    userMessage.modes[chatModes[0]] = { message: inputMessage };
    setListMessages([...listMessages, userMessage]);
    const chatbotMessageId = Date.now() + 1;
    const chatbotMessage: Messages = {
      id: chatbotMessageId,
      user: 'chatbot',
      datetime: new Date().toLocaleString(),
      isTyping: true,
      isLoading: true,
      modes: {},
      currentMode: chatModes[0],
    };
    setListMessages((prev) => [...prev, chatbotMessage]);
    try {
      const updateAgenticMessage = (mode: string, responseMode: ResponseMode) => {
        setActiveAgenticJobId(
          responseMode.agentic?.status && ['complete', 'failed', 'cancelled'].includes(responseMode.agentic.status)
            ? null
            : responseMode.agentic?.job_id ?? null
        );
        setListMessages((prev) =>
          prev.map((msg) =>
            msg.id === chatbotMessageId
              ? {
                  ...msg,
                  isLoading: !['complete', 'failed', 'cancelled'].includes(responseMode.agentic?.status ?? ''),
                  isTyping: false,
                  modes: { ...msg.modes, [mode]: responseMode },
                }
              : msg
          )
        );
      };
      const yearMin = agenticYearMin.trim() ? Number(agenticYearMin) : undefined;
      const yearMax = agenticYearMax.trim() ? Number(agenticYearMax) : undefined;
      const apiCalls = chatModes.map((mode) =>
        mode === chatModeLables['agentic rag']
          ? agenticChatAPI(inputMessage, {
              allowExternalWeb: agenticAllowExternalWeb,
              webProfileIds: agenticSelectedWebProfiles,
              corpora: agenticSelectedCorpora,
              geography: agenticGeography,
              yearMin: Number.isFinite(yearMin) ? yearMin : undefined,
              yearMax: Number.isFinite(yearMax) ? yearMax : undefined,
              numericMode: agenticNumericMode,
              onUpdate: (responseMode) => updateAgenticMessage(mode, responseMode),
            })
          : chatBotAPI(
              inputMessage,
              sessionId,
              model,
              mode,
              selectedFileNames?.map((f) => f.name)
            )
      );
      setInputMessage('');
      const results = await Promise.allSettled(apiCalls);
      results.forEach((result, index) => {
        const mode = chatModes[index];
        if (result.status === 'fulfilled') {
          // @ts-ignore
          if (result.value.response.data.status === 'Success') {
            const response = result.value.response.data.data;
            const responseMode: ResponseMode = {
              message: response.message,
              sources: response.info.sources,
              model: response.info.model,
              total_tokens: response.info.total_tokens,
              response_time: response.info.response_time,
              cypher_query: response.info.cypher_query,
              graphonly_entities: response.info.context ?? [],
              entities: response.info.entities ?? [],
              nodeDetails: response.info.nodedetails,
              error: response.info.error,
              metric_question: response.info?.metric_details?.question ?? '',
              metric_answer: response.info?.metric_details?.answer ?? '',
              metric_contexts: response.info?.metric_details?.contexts ?? '',
              agentic: response.info?.agentic,
            };
            if (index === 0) {
              simulateTypingEffect(chatbotMessageId, responseMode, mode, responseMode.message);
            } else {
              setListMessages((prev) =>
                prev.map((msg) =>
                  (msg.id === chatbotMessageId ? { ...msg, modes: { ...msg.modes, [mode]: responseMode } } : msg)
                )
              );
            }
          } else {
            const response = result.value.response.data;
            const responseMode: ResponseMode = {
              message: response.message,
              error: response.error,
            };
            if (index === 0) {
              simulateTypingEffect(chatbotMessageId, responseMode, response.data, responseMode.message);
            } else {
              setListMessages((prev) =>
                prev.map((msg) =>
                  (msg.id === chatbotMessageId ? { ...msg, modes: { ...msg.modes, [mode]: responseMode } } : msg)
                )
              );
            }
          }
        } else {
          console.error(`API call failed for mode ${mode}:`, result.reason);
          setListMessages((prev) =>
            prev.map((msg) =>
              (msg.id === chatbotMessageId
                ? {
                    ...msg,
                    modes: {
                      ...msg.modes,
                      [mode]: { message: 'Failed to fetch response for this mode.', error: result.reason },
                    },
                  }
                : msg)
            )
          );
        }
      });
      setListMessages((prev) =>
        prev.map((msg) => (msg.id === chatbotMessageId ? { ...msg, isLoading: false, isTyping: false } : msg))
      );
    } catch (error) {
      console.error('Error in handling chat:', error);
      if (error instanceof Error) {
        setListMessages((prev) =>
          prev.map((msg) =>
            (msg.id === chatbotMessageId
              ? {
                  ...msg,
                  isLoading: false,
                  isTyping: false,
                  modes: {
                    [chatModes[0]]: {
                      message: 'An error occurred while processing your request.',
                      error: error.message,
                    },
                  },
                }
              : msg)
          )
        );
      }
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => {
    scrollToBottom();
    setLoading(() => listMessages.some((msg) => msg.isLoading || msg.isTyping));
  }, [listMessages]);

  const handleCopy = (message: string, id: number) => {
    copy(message);
    setListMessages((msgs) =>
      msgs.map((msg) => {
        if (msg.id === id) {
          msg.copying = true;
        }
        return msg;
      })
    );
    setTimeout(() => {
      setListMessages((msgs) =>
        msgs.map((msg) => {
          if (msg.id === id) {
            msg.copying = false;
          }
          return msg;
        })
      );
    }, 2000);
  };

  const handleCancel = (id: number) => {
    cancel();
    setListMessages((msgs) => msgs.map((msg) => (msg.id === id ? { ...msg, speaking: false } : msg)));
  };

  const handleSpeak = (chatMessage: string, id: number) => {
    speak({ text: chatMessage }, typeof window !== 'undefined' && window.speechSynthesis != undefined);
    setListMessages((msgs) => {
      const messageWithSpeaking = msgs.find((msg) => msg.speaking);
      return msgs.map((msg) => (msg.id === id && !messageWithSpeaking ? { ...msg, speaking: true } : msg));
    });
  };

  const handleSwitchMode = (messageId: number, newMode: string) => {
    const activespeechId = listMessages.find((msg) => msg.speaking)?.id;
    if (speaking && messageId === activespeechId) {
      cancel();
      setListMessages((prev) =>
        prev.map((msg) => (msg.id === messageId ? { ...msg, currentMode: newMode, speaking: false } : msg))
      );
    } else {
      setListMessages((prev) => prev.map((msg) => (msg.id === messageId ? { ...msg, currentMode: newMode } : msg)));
    }
  };

  const detailsHandler = useCallback((chat: Messages, previousActiveChat: Messages | null) => {
    const currentMode = chat.modes[chat.currentMode];
    setModelModal(currentMode.model ?? '');
    setSourcesModal(currentMode.sources ?? []);
    setResponseTime(currentMode.response_time ?? 0);
    setTokensUsed(currentMode.total_tokens ?? 0);
    setcypherQuery(currentMode.cypher_query ?? '');
    setShowInfoModal(true);
    setChatsMode(chat.currentMode ?? '');
    setgraphEntitites(currentMode.graphonly_entities ?? []);
    setEntitiesModal(currentMode.entities ?? []);
    setmessageError(currentMode.error ?? '');
    setNodeDetailsModal(currentMode.nodeDetails ?? {});
    setMetricQuestion(currentMode.metric_question ?? '');
    setMetricContext(currentMode.metric_contexts ?? '');
    setMetricAnswer(currentMode.metric_answer ?? '');
    setActiveChat(chat);
    if (
      (previousActiveChat != null && chat.id != previousActiveChat?.id) ||
      (previousActiveChat != null && chat.currentMode != previousActiveChat.currentMode)
    ) {
      setNodes([]);
      setChunks([]);
      setInfoEntities([]);
      setMetricDetails(null);
    }
    if (previousActiveChat != null && chat.id != previousActiveChat?.id) {
      setMultiModelMetrics([]);
    }
  }, []);

  const speechHandler = useCallback((chat: Messages) => {
    if (chat.speaking) {
      handleCancel(chat.id);
    } else {
      handleSpeak(chat.modes[chat.currentMode]?.message, chat.id);
    }
  }, []);

  return (
    <div className='n-bg-palette-neutral-bg-weak flex! flex-col justify-between min-h-full max-h-full overflow-hidden relative'>
      {isDeleteChatLoading && (
        <div className='chatbot-deleteLoader'>
          <Loader title='Deleting...'></Loader>
        </div>
      )}
      <div
        className={`flex! overflow-y-auto pb-12 min-w-full pl-5 pr-5 chatBotContainer ${
          isChatOnly ? 'min-h-[calc(100dvh-114px)] max-h-[calc(100dvh-114px)]' : ''
        } `}
      >
        <Widget className='n-bg-palette-neutral-bg-weak w-full' header='' isElevated={false}>
          <div className='flex! flex-col gap-4 gap-y-4'>
            {listMessages.map((chat, index) => {
              const messagechatModes = Object.keys(chat.modes);
              return (
                <div
                  ref={messagesEndRef}
                  key={chat.id}
                  className={clsx(`flex! gap-2.5`, {
                    'flex-row': chat.user === 'chatbot',
                    'flex-row-reverse': chat.user !== 'chatbot',
                  })}
                >
                  <div className='w-8 h-8'>
                    {chat.user === 'chatbot' ? (
                      <Avatar
                        className='-ml-4'
                        hasStatus
                        name='KM'
                        size='large'
                        source={ChatBotAvatar}
                        status={connectionStatus ? 'online' : 'offline'}
                        type='image'
                        shape='square'
                      />
                    ) : (
                      <Avatar
                        className=''
                        hasStatus
                        name='KM'
                        size='large'
                        status={connectionStatus ? 'online' : 'offline'}
                        type='image'
                        shape='square'
                      />
                    )}
                  </div>
                  <Widget
                    header=''
                    isElevated={true}
                    className={`p-3! self-start ${isFullScreen ? 'max-w-[55%]' : ''} ${
                      chat.user === 'chatbot' ? 'n-bg-palette-neutral-bg-strong' : 'n-bg-palette-primary-bg-weak'
                    }`}
                  >
                    <div
                      className={`${
                        chat.isLoading && index === listMessages.length - 1 && chat.user === 'chatbot' ? 'loader' : ''
                      }`}
                    >
                      <div
                        className={
                          !isFullScreen
                            ? 'max-w-[250px] prose prose-sm sm:prose lg:prose-lg xl:prose-xl'
                            : 'prose prose-sm sm:prose lg:prose-lg xl:prose-xl max-w-none'
                        }
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw] as any}>
                          {chat.modes[chat.currentMode]?.message || ''}
                        </ReactMarkdown>
                      </div>
                      {chat.user === 'chatbot' && (
                        <AgenticDetailsBlock
                          details={chat.modes[chat.currentMode]?.agentic}
                          answerMarkdown={chat.modes[chat.currentMode]?.message}
                        />
                      )}
                    </div>
                    <div>
                      <div>
                        <Typography variant='body-small' className='pt-2 font-bold'>
                          {chat.datetime}
                        </Typography>
                      </div>
                      {chat.user === 'chatbot' &&
                        chat.id !== 2 &&
                        !chat.isLoading &&
                        !chat.isTyping &&
                        (!isFullScreen ? (
                          <Flex
                            flexDirection='row'
                            justifyContent={messagechatModes.length > 1 ? 'space-between' : 'unset'}
                            alignItems='center'
                          >
                            <CommonActions
                              chat={chat}
                              copyHandler={handleCopy}
                              detailsHandler={detailsHandler}
                              listMessages={listMessages}
                              speechHandler={speechHandler}
                              activeChat={activeChat}
                            ></CommonActions>
                            {messagechatModes.length > 1 && (
                              <ChatModesSwitch
                                currentMode={chat.currentMode}
                                switchToOtherMode={(index: number) => {
                                  const modes = Object.keys(chat.modes);
                                  const modeswtich = modes[index];
                                  handleSwitchMode(chat.id, modeswtich);
                                }}
                                isFullScreen={false}
                                currentModeIndex={messagechatModes.indexOf(chat.currentMode)}
                                modescount={messagechatModes.length}
                              />
                            )}
                          </Flex>
                        ) : (
                          <Flex flexDirection='row' justifyContent='space-between' alignItems='center'>
                            <Flex flexDirection='row' justifyContent='space-between' alignItems='center'>
                              <CommonActions
                                chat={chat}
                                copyHandler={handleCopy}
                                detailsHandler={detailsHandler}
                                listMessages={listMessages}
                                speechHandler={speechHandler}
                                activeChat={activeChat}
                              ></CommonActions>
                            </Flex>
                            <Box>
                              {messagechatModes.length > 1 && (
                                <ChatModesSwitch
                                  currentMode={chat.currentMode}
                                  switchToOtherMode={(index: number) => {
                                    const modes = Object.keys(chat.modes);
                                    const modeswtich = modes[index];
                                    handleSwitchMode(chat.id, modeswtich);
                                  }}
                                  isFullScreen={isFullScreen}
                                  currentModeIndex={messagechatModes.indexOf(chat.currentMode)}
                                  modescount={messagechatModes.length}
                                />
                              )}
                            </Box>
                          </Flex>
                        ))}
                    </div>
                  </Widget>
                </div>
              );
            })}
          </div>
        </Widget>
      </div>
      {isAgenticModeSelected && (
        <div className='n-bg-palette-neutral-bg-default border-t border-neutral-300 p-2 text-xs' data-testid='agentic-chat-settings'>
          <div className='mb-2 flex! flex-wrap gap-2 items-center'>
            <span className='font-bold'>Agentic RAG</span>
            {agenticExamples.map((example, index) => (
              <button
                key={example}
                type='button'
                className='rounded border px-2 py-1'
                onClick={() => setInputMessage(example)}
                title={example}
              >
                Пример {index + 1}
              </button>
            ))}
            {activeAgenticJobId && (
              <button type='button' className='rounded border px-2 py-1' onClick={cancelActiveAgenticJob}>
                Отменить job
              </button>
            )}
          </div>
          <div className='grid gap-2 md:grid-cols-2'>
            <div className='rounded bg-neutral-50 p-2'>
              <label className='flex gap-2 items-start'>
                <input
                  data-testid='agentic-chat-allow-web'
                  type='checkbox'
                  checked={agenticAllowExternalWeb}
                  onChange={(event) => setAgenticAllowExternalWeb(event.target.checked)}
                />
                <span>
                  Разрешить внешний Yandex Web Search. Во внешний контур уйдут только очищенные поисковые формулировки.
                </span>
              </label>
              {agenticAllowExternalWeb && (
                <div className='mt-2 grid grid-cols-2 gap-1'>
                  {agenticWebProfiles.map((profile) => (
                    <label key={profile.id} className='flex gap-1 items-center' title={profile.description}>
                      <input
                        type='checkbox'
                        checked={agenticSelectedWebProfiles.includes(profile.id)}
                        onChange={() => toggleAgenticWebProfile(profile.id)}
                      />
                      <span>
                        {profile.title}
                        {profile.metadata_only ? ' (metadata)' : ''}
                      </span>
                    </label>
                  ))}
                  <span className='opacity-70'>Максимум 2 профиля за итерацию.</span>
                </div>
              )}
            </div>
            <div className='rounded bg-neutral-50 p-2'>
              <div className='mb-1 font-bold'>Фильтры evidence</div>
              <div className='mb-2 flex! flex-wrap gap-2'>
                {agenticCorpusOptions.map((corpus) => (
                  <label key={corpus.id} className='flex gap-1 items-center'>
                    <input
                      type='checkbox'
                      checked={agenticSelectedCorpora.includes(corpus.id)}
                      onChange={() => toggleAgenticCorpus(corpus.id)}
                    />
                    <span>{corpus.label}</span>
                  </label>
                ))}
              </div>
              <div className='grid grid-cols-2 gap-2'>
                <input
                  className='rounded border p-1'
                  value={agenticGeography}
                  onChange={(event) => setAgenticGeography(event.target.value)}
                  placeholder='География: Russia / World'
                />
                <select
                  className='rounded border p-1'
                  value={agenticNumericMode}
                  onChange={(event) => setAgenticNumericMode(event.target.value as 'boost' | 'strict')}
                >
                  <option value='boost'>Числа: boost</option>
                  <option value='strict'>Числа: strict</option>
                </select>
                <input
                  className='rounded border p-1'
                  value={agenticYearMin}
                  onChange={(event) => setAgenticYearMin(event.target.value.replace(/[^\d]/g, '').slice(0, 4))}
                  placeholder='Год от'
                />
                <input
                  className='rounded border p-1'
                  value={agenticYearMax}
                  onChange={(event) => setAgenticYearMax(event.target.value.replace(/[^\d]/g, '').slice(0, 4))}
                  placeholder='Год до'
                />
              </div>
            </div>
          </div>
        </div>
      )}
      <div className='n-bg-palette-neutral-bg-weak flex! gap-2.5 bottom-0 p-2.5 w-full'>
        <form onSubmit={handleSubmit} className={`flex! gap-2.5 w-full ${!isFullScreen ? 'justify-between' : ''}`}>
          <TextInput
            className={`n-bg-palette-neutral-bg-default flex-grow-7 ${
              isFullScreen ? 'w-[calc(100%-105px)]' : 'w-[70%]'
            }`}
            value={inputMessage}
            isFluid
            onChange={handleInputChange}
            htmlAttributes={{
              type: 'text',
              'aria-label': 'chatbot-input',
              name: 'chatbot-input',
            }}
          />
          <SpotlightTarget id='chatbtn' hasPulse={true} indicatorVariant='border'>
            <ButtonWithToolTip
              label='Q&A Button'
              placement='top'
              text={`Ask a question.`}
              type='submit'
              disabled={loading || (!connectionStatus && !isAgenticModeSelected)}
              size='medium'
            >
              {buttonCaptions.ask}{' '}
              {selectedFileNames != undefined && selectedFileNames.length > 0 && `(${selectedFileNames.length})`}
            </ButtonWithToolTip>
          </SpotlightTarget>
        </form>
      </div>
      <Suspense fallback={<FallBackDialog />}>
        <Modal
          modalProps={{
            id: 'retrieval-information',
            className: 'n-p-token-4 n-bg-palette-neutral-bg-weak n-rounded-lg',
          }}
          onClose={() => setShowInfoModal(false)}
          isOpen={showInfoModal}
          size={'large'}
        >
          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
            <IconButton
              size='large'
              htmlAttributes={{
                title: 'download chat info',
              }}
              isClean
              ariaLabel='download chat info'
              isDisabled={metricsLoading || infoLoading}
              onClick={() => {
                downloadClickHandler(
                  {
                    chatResponse: activeChat,
                    chunks,
                    metricDetails,
                    communities,
                    responseTime,
                    entities: infoEntities,
                    nodes,
                    tokensUsed,
                    model,
                    multiModelMetrics,
                  },
                  downloadLinkRef,
                  'graph-builder-chat-details.json'
                );
              }}
            >
              <ArrowDownTrayIconOutline className='n-size-token-7' />
              <TextLink ref={downloadLinkRef} className='hidden!'>
                ""
              </TextLink>
            </IconButton>
            <IconButton
              size='large'
              htmlAttributes={{
                title: 'close pop up',
              }}
              ariaLabel='close pop up'
              isClean
              onClick={() => setShowInfoModal(false)}
            >
              <XMarkIconOutline className='n-size-token-7' />
            </IconButton>
          </div>
          <InfoModal
            sources={sourcesModal}
            model={modelModal}
            entities_ids={entitiesModal}
            response_time={responseTime}
            total_tokens={tokensUsed}
            mode={chatsMode}
            cypher_query={cypherQuery}
            graphonly_entities={graphEntitites}
            error={messageError}
            nodeDetails={nodeDetailsModal}
            metricanswer={metricAnswer}
            metriccontexts={metricContext}
            metricquestion={metricQuestion}
            metricmodel={model}
            nodes={nodes}
            infoEntities={infoEntities}
            relationships={relationships}
            chunks={chunks}
            metricDetails={activeChat != undefined && metricDetails != null ? metricDetails : undefined}
            metricError={activeChat != undefined && metricDetails != null ? (metricDetails.error as string) : ''}
            communities={communities}
            infoLoading={infoLoading}
            metricsLoading={metricsLoading}
            saveInfoEntitites={saveInfoEntitites}
            saveChatRelationships={saveChatRelationships}
            saveChunks={saveChunks}
            saveCommunities={saveCommunities}
            saveMetrics={saveMetrics}
            saveNodes={saveNodes}
            toggleInfoLoading={toggleInfoLoading}
            toggleMetricsLoading={toggleMetricsLoading}
            saveMultimodemetrics={saveMultimodemetrics}
            activeChatmodes={activeChat?.modes}
            multiModelMetrics={multiModelMetrics}
          />
        </Modal>
      </Suspense>
    </div>
  );
};

export default Chatbot;
