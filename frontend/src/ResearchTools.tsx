import { useCallback, useEffect, useState, type FormEvent } from 'react';

import { apiFetch, getApiBaseUrl } from './lib/api';
import type { Paper, PaperListResponse } from './types';

type SavedAnalysis = {
  id: string;
  title: string;
  paper_ids: string[];
  mode: string;
  created_at: string;
};

type Citation = {
  title?: string;
  page?: number | null;
};

type ComparisonCell = {
  paper_id: string;
  text?: string | null;
  status: string;
  citations: Citation[];
  warnings: string[];
};

type Comparison = SavedAnalysis & {
  model?: string | null;
  columns: Array<{ paper_id: string; title: string; year?: number | null }>;
  rows: Array<{ field: string; label: string; cells: ComparisonCell[] }>;
  observations: Array<{ text: string; cites: Array<{ paper_title: string; field: string }> }>;
  warnings: string[];
};

type GapPassage = {
  paper_title: string;
  page?: number | null;
  sentence: string;
};

type ResearchGap = {
  id: number;
  description: string;
  research_question?: string | null;
  experimental_direction?: string | null;
  source: string;
  kind: string;
  evidence_strength: string;
  supporting_papers: Array<{ paper_id: string; title: string }>;
  passages: GapPassage[];
  status: string;
};

type GapAnalysis = SavedAnalysis & {
  disclaimer: string;
  model?: string | null;
  stats: Record<string, number>;
  frequent_terms: Array<{ term: string; count?: number }>;
  distinctive_terms: Array<{ term: string; paper_title?: string }>;
  paper_keywords: Array<{ title: string; keywords: string[] }>;
  topics: Array<{ id: number; terms: string[] }>;
  gaps: ResearchGap[];
  warnings: string[];
};

function eligiblePapers(papers: Paper[]) {
  return papers.filter((paper) => paper.processing_status === 'completed');
}

function PaperPicker({
  papers,
  selected,
  onChange,
  max,
}: {
  papers: Paper[];
  selected: string[];
  onChange: (ids: string[]) => void;
  max: number;
}) {
  const readyPapers = eligiblePapers(papers);
  return (
    <div className="analysis-paper-picker">
      {readyPapers.length ? readyPapers.map((paper) => {
        const checked = selected.includes(paper.id);
        const disabled = !checked && selected.length >= max;
        return (
          <label className="analysis-paper-option" key={paper.id}>
            <input
              type="checkbox"
              checked={checked}
              disabled={disabled}
              onChange={(event) => {
                onChange(event.target.checked
                  ? [...selected, paper.id]
                  : selected.filter((id) => id !== paper.id));
              }}
            />
            <span>
              <strong>{paper.title}</strong>
              <small>{paper.authors.join(', ') || 'Author unknown'}{paper.year ? ` · ${paper.year}` : ''}</small>
            </span>
          </label>
        );
      }) : <p className="muted">No processed papers with extracted text yet. Upload and finish processing a PDF first.</p>}
    </div>
  );
}

async function downloadExport(path: string, token: string, fileName: string) {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error('Could not download the export.');
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ComparisonPage({ token }: { token: string }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [saved, setSaved] = useState<SavedAnalysis[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [title, setTitle] = useState('');
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const reload = useCallback(async () => {
    const [paperResponse, savedResponse] = await Promise.all([
      apiFetch<PaperListResponse>('/papers?page=1&page_size=100', {}, token),
      apiFetch<SavedAnalysis[]>('/comparisons', {}, token),
    ]);
    setPapers(paperResponse.items);
    setSaved(savedResponse);
  }, [token]);

  useEffect(() => {
    void reload().catch((err: unknown) => setError(err instanceof Error ? err.message : 'Unable to load comparisons'));
  }, [reload]);

  const createComparison = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedIds.length < 2) return;
    setBusy(true);
    setError('');
    try {
      const result = await apiFetch<Comparison>('/comparisons', {
        method: 'POST',
        body: JSON.stringify({ paper_ids: selectedIds, title: title.trim() || undefined }),
      }, token);
      setComparison(result);
      setTitle('');
      setSelectedIds([]);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to compare papers');
    } finally {
      setBusy(false);
    }
  };

  const openSaved = async (id: string) => {
    setBusy(true);
    setError('');
    try {
      setComparison(await apiFetch<Comparison>(`/comparisons/${id}`, {}, token));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to open comparison');
    } finally {
      setBusy(false);
    }
  };

  const removeSaved = async (id: string) => {
    if (!window.confirm('Delete this saved comparison?')) return;
    setError('');
    try {
      await apiFetch<void>(`/comparisons/${id}`, { method: 'DELETE' }, token);
      if (comparison?.id === id) setComparison(null);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete comparison');
    }
  };

  const exportComparison = async (format: 'markdown' | 'json') => {
    if (!comparison) return;
    setError('');
    try {
      await downloadExport(`/comparisons/${comparison.id}/export?format=${format}`, token,
        format === 'json' ? 'comparison.json' : 'comparison.md');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to export comparison');
    }
  };

  return (
    <div className="research-tool-layout">
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <h2>Compare papers</h2>
            <p className="muted">Compare extracted evidence across 2–5 processed papers.</p>
          </div>
        </div>
        <form className="stack-form" onSubmit={createComparison}>
          <label>
            <span>Report title (optional)</span>
            <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Approaches to retrieval evaluation" />
          </label>
          <PaperPicker papers={papers} selected={selectedIds} onChange={setSelectedIds} max={5} />
          <button className="primary-button" type="submit" disabled={busy || selectedIds.length < 2}>
            {busy ? 'Comparing…' : `Compare ${selectedIds.length || ''} papers`}
          </button>
        </form>

        <div className="saved-analysis-list">
          <h3>Saved comparisons</h3>
          {saved.length ? saved.map((item) => (
            <div className="saved-analysis-row" key={item.id}>
              <button type="button" className="saved-analysis-open" onClick={() => void openSaved(item.id)}>
                <strong>{item.title}</strong>
                <small>{item.mode} · {new Date(item.created_at).toLocaleDateString()}</small>
              </button>
              <button type="button" className="danger-button" onClick={() => void removeSaved(item.id)} aria-label={`Delete ${item.title}`}>Delete</button>
            </div>
          )) : <p className="muted">No saved comparisons yet.</p>}
        </div>
      </section>

      <section className="panel-card analysis-output">
        {error && <div className="error-box">{error}</div>}
        {comparison ? (
          <>
            <div className="panel-header">
              <div>
                <p className="eyebrow">{comparison.mode === 'llm' ? comparison.model ?? 'AI analysis' : 'Extractive analysis'}</p>
                <h2>{comparison.title}</h2>
              </div>
              <div className="export-actions">
                <button type="button" className="secondary-button" onClick={() => void exportComparison('markdown')}>Export MD</button>
                <button type="button" className="secondary-button" onClick={() => void exportComparison('json')}>Export JSON</button>
              </div>
            </div>
            {comparison.warnings.map((warning, index) => <div className="warning-box" key={index}>{warning}</div>)}
            <div className="table-scroll">
              <table className="comparison-table">
                <thead>
                  <tr><th>Topic</th>{comparison.columns.map((column) => <th key={column.paper_id}>{column.title}{column.year ? ` (${column.year})` : ''}</th>)}</tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row) => (
                    <tr key={row.field}>
                      <th>{row.label}</th>
                      {row.cells.map((cell, index) => (
                        <td key={`${row.field}-${cell.paper_id}-${index}`}>
                          <span className={`status-badge neutral cell-status ${cell.status}`}>{cell.status}</span>
                          <p>{cell.text || 'Not found in the paper text.'}</p>
                          {cell.citations.map((citation, citationIndex) => (
                            <small className="citation-line" key={citationIndex}>
                              {citation.title ?? 'Source'}{citation.page ? ` · p. ${citation.page}` : ''}
                            </small>
                          ))}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {comparison.observations.length > 0 && (
              <div className="analysis-subsection">
                <h3>Cross-paper observations</h3>
                {comparison.observations.map((observation, index) => (
                  <article className="observation-card" key={index}>
                    <p>{observation.text}</p>
                    <small>{observation.cites.map((cite) => `${cite.paper_title} · ${cite.field}`).join(' / ')}</small>
                  </article>
                ))}
              </div>
            )}
          </>
        ) : <div className="empty-state">Choose at least two processed papers to create a comparison, or open a saved one.</div>}
      </section>
    </div>
  );
}

export function GapAnalysisPage({ token }: { token: string }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [saved, setSaved] = useState<SavedAnalysis[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [title, setTitle] = useState('');
  const [analysis, setAnalysis] = useState<GapAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const reload = useCallback(async () => {
    const [paperResponse, savedResponse] = await Promise.all([
      apiFetch<PaperListResponse>('/papers?page=1&page_size=100', {}, token),
      apiFetch<SavedAnalysis[]>('/gaps', {}, token),
    ]);
    setPapers(paperResponse.items);
    setSaved(savedResponse);
  }, [token]);

  useEffect(() => {
    void reload().catch((err: unknown) => setError(err instanceof Error ? err.message : 'Unable to load gap analyses'));
  }, [reload]);

  const analyze = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedIds.length < 2) return;
    setBusy(true);
    setError('');
    try {
      const result = await apiFetch<GapAnalysis>('/gaps/analyze', {
        method: 'POST',
        body: JSON.stringify({ paper_ids: selectedIds, title: title.trim() || undefined }),
      }, token);
      setAnalysis(result);
      setTitle('');
      setSelectedIds([]);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to analyze research gaps');
    } finally {
      setBusy(false);
    }
  };

  const openSaved = async (id: string) => {
    setBusy(true);
    setError('');
    try {
      setAnalysis(await apiFetch<GapAnalysis>(`/gaps/${id}`, {}, token));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to open gap analysis');
    } finally {
      setBusy(false);
    }
  };

  const removeSaved = async (id: string) => {
    if (!window.confirm('Delete this saved gap analysis?')) return;
    setError('');
    try {
      await apiFetch<void>(`/gaps/${id}`, { method: 'DELETE' }, token);
      if (analysis?.id === id) setAnalysis(null);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete gap analysis');
    }
  };

  return (
    <div className="research-tool-layout">
      <section className="panel-card">
        <h2>Research gap analysis</h2>
        <p className="muted">Find candidate themes from limitations and future-work statements in your selected papers.</p>
        <form className="stack-form" onSubmit={analyze}>
          <label>
            <span>Report title (optional)</span>
            <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Open questions in paper retrieval" />
          </label>
          <PaperPicker papers={papers} selected={selectedIds} onChange={setSelectedIds} max={30} />
          <button className="primary-button" type="submit" disabled={busy || selectedIds.length < 2}>
            {busy ? 'Analyzing…' : `Analyze ${selectedIds.length || ''} papers`}
          </button>
        </form>
        <div className="saved-analysis-list">
          <h3>Saved analyses</h3>
          {saved.length ? saved.map((item) => (
            <div className="saved-analysis-row" key={item.id}>
              <button type="button" className="saved-analysis-open" onClick={() => void openSaved(item.id)}>
                <strong>{item.title}</strong>
                <small>{item.mode} · {new Date(item.created_at).toLocaleDateString()}</small>
              </button>
              <button type="button" className="danger-button" onClick={() => void removeSaved(item.id)} aria-label={`Delete ${item.title}`}>Delete</button>
            </div>
          )) : <p className="muted">No saved analyses yet.</p>}
        </div>
      </section>

      <section className="panel-card analysis-output">
        {error && <div className="error-box">{error}</div>}
        {analysis ? (
          <>
            <div className="panel-header">
              <div>
                <p className="eyebrow">{analysis.mode === 'llm' ? analysis.model ?? 'AI analysis' : 'Extractive analysis'}</p>
                <h2>{analysis.title}</h2>
              </div>
              <span className="status-badge neutral">{analysis.stats.gaps_returned ?? analysis.gaps.length} candidate gaps</span>
            </div>
            <div className="warning-box">{analysis.disclaimer}</div>
            <div className="stats-grid compact-stats">
              <div><small>Papers</small><strong>{analysis.stats.papers ?? 0}</strong></div>
              <div><small>Statements</small><strong>{analysis.stats.statements ?? 0}</strong></div>
              <div><small>Text chunks</small><strong>{analysis.stats.chunks ?? 0}</strong></div>
            </div>
            {analysis.warnings.map((warning, index) => <div className="warning-box" key={index}>{warning}</div>)}
            <div className="gap-list">
              {analysis.gaps.map((gap) => (
                <article className="gap-card" key={gap.id}>
                  <div className="panel-header">
                    <h3>Candidate gap {gap.id}</h3>
                    <span className={`status-badge neutral strength-${gap.evidence_strength}`}>{gap.evidence_strength} evidence</span>
                  </div>
                  <p>{gap.description}</p>
                  {gap.research_question && <div><strong>Research question</strong><p>{gap.research_question}</p></div>}
                  {gap.experimental_direction && <div><strong>Possible direction</strong><p>{gap.experimental_direction}</p></div>}
                  <p className="muted">{gap.status}</p>
                  <div className="supporting-paper-list">
                    <strong>Supporting papers</strong>
                    {gap.supporting_papers.map((paper) => <span className="tag" key={paper.paper_id}>{paper.title}</span>)}
                  </div>
                  {gap.passages.map((passage, index) => (
                    <blockquote key={index}>
                      “{passage.sentence}”
                      <cite>{passage.paper_title}{passage.page ? ` · p. ${passage.page}` : ''}</cite>
                    </blockquote>
                  ))}
                </article>
              ))}
              {!analysis.gaps.length && <div className="empty-state">No limitation or future-work statements were detected in this set.</div>}
            </div>
          </>
        ) : <div className="empty-state">Select at least two processed papers to analyze candidate research gaps, or open a saved analysis.</div>}
      </section>
    </div>
  );
}
