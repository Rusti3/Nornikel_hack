import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react';
import DemoHeader from '../DemoShell/DemoHeader';
import { cancelIngestJob, listIngestJobs, startIngestJob } from '../../services/MEKGAPI';
import { IngestJob, ingestProgressText, waitForIngestJob } from '../../services/IngestAPI';
import './DataUploadPage.css';

const MAX_SIZE = 100 * 1024 * 1024;
const ACCEPTED = ['.pdf', '.doc', '.docx', '.docm', '.pptx', '.xls', '.xlsx'];

const validFiles = (values: File[]) => values
  .filter((file) => ACCEPTED.some((suffix) => file.name.toLowerCase().endsWith(suffix)))
  .filter((file) => file.size > 0 && file.size <= MAX_SIZE)
  .slice(0, 10);

export default function DataUploadPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [category, setCategory] = useState('auto');
  const [dragging, setDragging] = useState(false);
  const [active, setActive] = useState<IngestJob | null>(null);
  const [history, setHistory] = useState<IngestJob[]>([]);
  const [error, setError] = useState('');

  const refresh = async () => {
    try {
      const response = await listIngestJobs();
      setHistory(response.data.items ?? []);
    } catch {
      setHistory([]);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const addFiles = (values: File[]) => {
    const accepted = validFiles(values);
    if (accepted.length !== values.length) {
      setError('Часть файлов отклонена: поддерживаются PDF/Office, максимум 100 МБ и 10 файлов.');
    } else {
      setError('');
    }
    setFiles((current) => {
      const map = new Map(current.map((file) => [`${file.name}:${file.size}`, file]));
      accepted.forEach((file) => map.set(`${file.name}:${file.size}`, file));
      return Array.from(map.values()).slice(0, 10);
    });
  };

  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  };

  const upload = async () => {
    if (!files.length) return;
    setError('');
    try {
      const accepted = await startIngestJob(files, { category });
      const job: IngestJob = { job_id: accepted.data.job_id, status: accepted.data.status };
      setActive(job);
      const finished = await waitForIngestJob(job.job_id, setActive);
      setActive(finished);
      if (finished.status === 'complete') setFiles([]);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось запустить индексацию');
    }
  };

  const cancel = async () => {
    if (!active?.job_id) return;
    await cancelIngestJob(active.job_id);
  };

  const event = active?.state?.last_event ?? {};
  const progress = Math.max(0, Math.min(1, Number(event.progress ?? active?.state?.progress ?? 0)));

  return (
    <div className='data-page'>
      <DemoHeader />
      <main className='data-main'>
        <section className='data-hero'>
          <div className='data-kicker'>DATA PIPELINE</div>
          <h1>Добавить новые знания</h1>
          <p>Файл автоматически попадёт в BM25, vector search и граф Neo4j.</p>
        </section>

        <section className='data-upload-card'>
          <div
            className={`data-dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={drop}
            onClick={() => inputRef.current?.click()}
            role='button'
            tabIndex={0}
          >
            <input
              ref={inputRef}
              type='file'
              multiple
              hidden
              accept={ACCEPTED.join(',')}
              onChange={(event: ChangeEvent<HTMLInputElement>) => addFiles(Array.from(event.target.files ?? []))}
            />
            <div className='data-drop-icon'>＋</div>
            <strong>Перетащите документы или нажмите для выбора</strong>
            <span>PDF, Word, PowerPoint, Excel · до 100 МБ</span>
          </div>

          {files.length > 0 && (
            <div className='data-file-list'>
              {files.map((file) => (
                <div className='data-file' key={`${file.name}:${file.size}`}>
                  <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(1)} МБ</span></div>
                  <button onClick={() => setFiles((current) => current.filter((item) => item !== file))}>Удалить</button>
                </div>
              ))}
            </div>
          )}

          <div className='data-upload-actions'>
            <label>
              Корпус
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value='auto'>Авто</option>
                <option value='internal_reports'>Доклады</option>
                <option value='scientific_journals'>Журналы</option>
                <option value='conference_materials'>Материалы конференций</option>
                <option value='reviews'>Обзоры</option>
                <option value='scientific_articles'>Статьи</option>
              </select>
            </label>
            <button className='data-upload-button' disabled={!files.length || Boolean(active && !['complete','failed','cancelled'].includes(active.status))} onClick={() => void upload()}>
              Отправить в pipeline
            </button>
          </div>

          {active && (
            <div className='data-progress'>
              <div className='data-progress-head'><strong>{active.state?.current_file || 'Индексация'}</strong><span>{Math.round(progress * 100)}%</span></div>
              <div className='data-progress-track'><span style={{ width: `${Math.max(2, progress * 100)}%` }} /></div>
              <div className='data-progress-text'>{ingestProgressText(active).replace(/[#`]/g, '')}</div>
              {!['complete','failed','cancelled'].includes(active.status) && <button className='data-cancel' onClick={() => void cancel()}>Остановить</button>}
              {active.error && <div className='data-error'>{active.error}</div>}
            </div>
          )}
          {error && <div className='data-error'>{error}</div>}
        </section>

        <section className='data-history'>
          <div className='data-history-head'><h2>Последние загрузки</h2><button onClick={() => void refresh()}>Обновить</button></div>
          <div className='data-history-list'>
            {history.map((job) => (
              <div className='data-history-row' key={job.job_id}>
                <div><strong>{job.state?.current_file || job.result?.documents?.[0]?.file_name || 'Пакет документов'}</strong><span>{job.job_id}</span></div>
                <span className={`data-status status-${job.status}`}>{job.result?.outcome || job.state?.stage || job.status}</span>
              </div>
            ))}
            {!history.length && <div className='data-empty'>Загрузок пока нет.</div>}
          </div>
        </section>
      </main>
    </div>
  );
}
