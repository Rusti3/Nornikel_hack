import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import DemoHeader from '../DemoShell/DemoHeader';
import { getGraphPathways } from '../../services/MEKGAPI';
import { nvlOptions } from '../../utils/Constants';
import './GraphPathwaysPage.css';

type PathNode = {
  id?: string;
  name?: string;
  canonical_name?: string;
  property_name?: string;
  text?: string;
  labels?: string[];
  confidence?: number;
  validation_status?: string;
};

type Pathway = {
  id: string;
  experiment: PathNode;
  material?: PathNode | null;
  process?: PathNode | null;
  equipment?: PathNode | null;
  result?: PathNode | null;
  result_title?: string | null;
  missing_stages: string[];
  complete: boolean;
  confidence: number;
  status: 'verified' | 'low_confidence' | 'conflicting';
  evidence?: {
    page?: number;
    slide?: number;
    text?: string;
    document?: { id?: string; file_name?: string; category?: string };
  };
};

type PathwayResponse = {
  pathways: Pathway[];
  nodes: Array<Record<string, any>>;
  relationships: Array<Record<string, any>>;
  coverage: { total: number; complete: number; incomplete: number };
};

const title = (node?: PathNode | null) =>
  node?.name || node?.canonical_name || node?.property_name || node?.text || 'Связь не извлечена';

const roleNames: Record<string, string> = {
  material: 'Материал', process: 'Процесс', equipment: 'Оборудование', result: 'Результат',
};

const roleColors: Record<string, string> = {
  material: '#2563eb', process: '#7c3aed', equipment: '#d97706', result: '#059669', experiment: '#475467',
};

export default function GraphPathwaysPage() {
  const [params] = useSearchParams();
  const agentJobId = params.get('agent_job_id') || undefined;
  const [query, setQuery] = useState('');
  const [includeIncomplete, setIncludeIncomplete] = useState(true);
  const [mode, setMode] = useState<'chains' | 'network'>('chains');
  const [data, setData] = useState<PathwayResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async (search = query) => {
    setLoading(true);
    setError('');
    try {
      const response = await getGraphPathways({
        query: search.trim() || undefined,
        agent_job_id: agentJobId,
        include_incomplete: includeIncomplete,
        limit: 60,
      });
      setData(response.data as PathwayResponse);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось загрузить графовые цепочки');
    } finally {
      setLoading(false);
    }
  }, [agentJobId, includeIncomplete, query]);

  useEffect(() => { void load(''); }, [agentJobId, includeIncomplete]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void load();
  };

  const networkNodes = useMemo(() => (data?.nodes ?? []).map((node) => ({
    id: node.id,
    caption: node.title,
    labels: node.labels,
    properties: node.properties,
    size: node.role === 'experiment' ? 28 : 34,
    color: roleColors[node.role] ?? '#667085',
    captionAlign: 'bottom',
  })), [data]);
  const networkRels = useMemo(() => (data?.relationships ?? []).map((relationship) => ({
    id: relationship.id,
    from: relationship.from,
    to: relationship.to,
    caption: relationship.type,
  })), [data]);

  return (
    <div className='pathways-page'>
      <DemoHeader />
      <main className='pathways-main'>
        <section className='pathways-hero'>
          <div>
            <div className='pathways-kicker'>KNOWLEDGE GRAPH</div>
            <h1>Технологические цепочки</h1>
            <p>Только связи, подтверждённые извлечёнными Experiment и исходными документами.</p>
          </div>
          <div className='pathways-stats'>
            <strong>{data?.coverage?.complete ?? 0}</strong><span>полных</span>
            <strong>{data?.coverage?.incomplete ?? 0}</strong><span>неполных</span>
          </div>
        </section>

        <section className='pathways-toolbar'>
          <form onSubmit={submit} className='pathways-search'>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder='Например: никель, выщелачивание, SO₂…' />
            <button type='submit' disabled={loading}>Найти</button>
          </form>
          <label className='pathways-check'>
            <input type='checkbox' checked={includeIncomplete} onChange={(event) => setIncludeIncomplete(event.target.checked)} />
            Показывать пробелы
          </label>
          <div className='pathways-mode'>
            <button className={mode === 'chains' ? 'active' : ''} onClick={() => setMode('chains')}>Цепочки</button>
            <button className={mode === 'network' ? 'active' : ''} onClick={() => setMode('network')}>Сеть</button>
          </div>
        </section>

        {agentJobId && <div className='pathways-context'>Граф отфильтрован по evidence ответа <code>{agentJobId}</code>.</div>}
        {error && <div className='pathways-error'>{error}</div>}
        {loading && <div className='pathways-loading'>Собираю подтверждённые цепочки…</div>}

        {!loading && mode === 'chains' && (
          <section className='pathways-list'>
            {(data?.pathways ?? []).map((pathway) => (
              <article key={pathway.id} className={`pathway-card status-${pathway.status}`}>
                <div className='pathway-card-head'>
                  <div>
                    <strong>{title(pathway.experiment)}</strong>
                    <span>{pathway.evidence?.document?.file_name || 'Источник не установлен'}</span>
                  </div>
                  <div className='pathway-confidence'>confidence {pathway.confidence.toFixed(2)}</div>
                </div>
                <div className='pathway-chain'>
                  {(['material', 'process', 'equipment', 'result'] as const).map((role, index) => {
                    const node = pathway[role];
                    const value = role === 'result' ? pathway.result_title || title(node) : title(node);
                    return (
                      <div className='pathway-step-wrap' key={role}>
                        <div className={`pathway-step ${node?.id ? '' : 'missing'}`}>
                          <span>{roleNames[role]}</span>
                          <strong>{value}</strong>
                        </div>
                        {index < 3 && <div className={`pathway-arrow ${node?.id ? '' : 'missing'}`}>→</div>}
                      </div>
                    );
                  })}
                </div>
                <div className='pathway-evidence'>
                  {pathway.evidence?.page ? `стр. ${pathway.evidence.page}` : ''}
                  {pathway.evidence?.slide ? `слайд ${pathway.evidence.slide}` : ''}
                  {pathway.missing_stages.length > 0 && ` · пробелы: ${pathway.missing_stages.map((item) => roleNames[item] || item).join(', ')}`}
                </div>
              </article>
            ))}
            {!data?.pathways?.length && <div className='pathways-empty'>По заданному фильтру цепочки не найдены.</div>}
          </section>
        )}

        {!loading && mode === 'network' && (
          <section className='pathways-network'>
            {networkNodes.length ? (
              <InteractiveNvlWrapper
                nodes={networkNodes as any}
                rels={networkRels as any}
                nvlOptions={{ ...nvlOptions, instanceId: 'pathways-network' }}
                interactionOptions={{ selectOnClick: true }}
              />
            ) : <div className='pathways-empty'>Нет узлов для отображения.</div>}
          </section>
        )}
      </main>
    </div>
  );
}
