import { useEffect, useState, type ChangeEvent, type FormEvent } from 'react';
import { BrowserRouter, NavLink, Navigate, Outlet, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import { BookOpenText, ChartNoAxesCombined, Copy, FileText, GitCompareArrows, LogOut, MessageSquareText, Search, Sparkles, Trash2, Upload, User } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

import { AuthProvider, useAuth } from './context/AuthContext';
import { apiFetch, getApiBaseUrl, loginRequest, registerRequest } from './lib/api';
import { ComparisonPage, GapAnalysisPage } from './ResearchTools';
import type { ChatSession, ChatSessionDetail, Paper, PaperListResponse, SearchResponse } from './types';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route element={<ProtectedLayout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/papers" element={<LibraryPage />} />
            <Route path="/papers/upload" element={<UploadPage />} />
            <Route path="/papers/:paperId" element={<PaperDetailPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/compare" element={<ComparisonRoute />} />
            <Route path="/gaps" element={<GapsRoute />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

function ProtectedLayout() {
  const { user, signOut, isAuthenticated, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div className="empty-state">Loading your workspace…</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">R</div>
          <div>
            <strong>Research Papers</strong>
            <small>Library workspace</small>
          </div>
        </div>

        <nav className="nav">
          <NavLink to="/dashboard" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <BookOpenText size={18} /> Dashboard
          </NavLink>
          <NavLink to="/papers" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <FileText size={18} /> My Papers
          </NavLink>
          <NavLink to="/papers/upload" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <Upload size={18} /> Upload PDF
          </NavLink>
          <NavLink to="/search" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <Search size={18} /> Search
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <MessageSquareText size={18} /> Chat
          </NavLink>
          <NavLink to="/compare" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <GitCompareArrows size={18} /> Compare papers
          </NavLink>
          <NavLink to="/gaps" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <ChartNoAxesCombined size={18} /> Research gaps
          </NavLink>
        </nav>

        <div className="sidebar-footer">
          <div className="user-chip">
            <User size={16} />
            <div>
              <strong>{user?.full_name ?? 'User'}</strong>
              <span>{user?.email ?? 'user@example.com'}</span>
            </div>
          </div>

          <button
            type="button"
            className="ghost-button"
            onClick={signOut}
          >
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </aside>

      <main className="content-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Workspace</p>
            <h1>{pageTitle(location.pathname)}</h1>
          </div>
        </header>
        <Outlet />
      </main>
    </div>
  );
}

function pageTitle(pathname: string) {
  if (pathname.startsWith('/papers/upload')) return 'Upload PDF';
  if (pathname.startsWith('/papers/')) return 'Paper details';
  if (pathname.startsWith('/search')) return 'Search library';
  if (pathname.startsWith('/chat')) return 'Research chat';
  if (pathname.startsWith('/compare')) return 'Compare papers';
  if (pathname.startsWith('/gaps')) return 'Research gap analysis';
  if (pathname.startsWith('/papers')) return 'My papers';
  return 'Dashboard';
}

function ComparisonRoute() {
  const { token } = useAuth();
  return token ? <ComparisonPage token={token} /> : null;
}

function GapsRoute() {
  const { token } = useAuth();
  return token ? <GapAnalysisPage token={token} /> : null;
}

function LoginPage() {
  const navigate = useNavigate();
  const { signIn, isAuthenticated } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      const tokens = await loginRequest(email.trim(), password);
      await signIn(tokens);
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to sign in');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-header">
          <div className="brand-mark large">R</div>
          <h1>Welcome back</h1>
          <p>Access your research library and AI assistant.</p>
        </div>

        <form onSubmit={onSubmit} className="stack-form">
          {error && <div className="error-box">{error}</div>}
          <label>
            <span>Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>
          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <p className="auth-footer">
          Need an account? <NavLink to="/register">Create one</NavLink>
        </p>
      </div>
    </div>
  );
}

function RegisterPage() {
  const navigate = useNavigate();
  const { signIn, isAuthenticated } = useAuth();
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      const tokens = await registerRequest(fullName.trim(), email.trim(), password);
      await signIn(tokens);
      navigate('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-header">
          <div className="brand-mark large">R</div>
          <h1>Create your account</h1>
          <p>Start collecting and exploring your paper library.</p>
        </div>

        <form onSubmit={onSubmit} className="stack-form">
          {error && <div className="error-box">{error}</div>}
          <label>
            <span>Full name</span>
            <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} required />
          </label>
          <label>
            <span>Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
          </label>
          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account? <NavLink to="/login">Sign in</NavLink>
        </p>
      </div>
    </div>
  );
}

function DashboardPage() {
  const { token } = useAuth();
  const [papers, setPapers] = useState<PaperListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!token) return;

    void (async () => {
      try {
        setLoading(true);
        const response = await apiFetch<PaperListResponse>('/papers?page=1&page_size=8', {}, token);
        setPapers(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load papers');
      } finally {
        setLoading(false);
      }
    })();
  }, [token]);

  const stats = papers ? {
    total: papers.total,
    unread: papers.items.filter((paper) => paper.reading_status === 'unread').length,
    reading: papers.items.filter((paper) => paper.reading_status === 'reading').length,
    processed: papers.items.filter((paper) => paper.processing_status === 'completed').length,
  } : { total: 0, unread: 0, reading: 0, processed: 0 };

  return (
    <div className="page-grid">
      <div className="stats-grid">
        <StatCard label="Total papers" value={String(stats.total)} accent="blue" />
        <StatCard label="Unread" value={String(stats.unread)} accent="amber" />
        <StatCard label="In progress" value={String(stats.reading)} accent="green" />
        <StatCard label="Processed" value={String(stats.processed)} accent="purple" />
      </div>

      <div className="panel-card">
        <div className="panel-header">
          <h2>Recent library</h2>
          <NavLink to="/papers" className="text-link">View all</NavLink>
        </div>

        {error && <div className="error-box">{error}</div>}

        {loading ? (
          <div className="empty-state">Loading papers…</div>
        ) : (
          <div className="list-stack">
            {papers?.items.length ? papers.items.map((paper) => (
              <PaperSummaryRow key={paper.id} paper={paper} />
            )) : <div className="empty-state">No papers yet. Upload your first PDF.</div>}
          </div>
        )}
      </div>
    </div>
  );
}

function LibraryPage() {
  const { token } = useAuth();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState('all');
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!token) return;

    void (async () => {
      try {
        setLoading(true);
        const params = new URLSearchParams({ page: String(page), page_size: '12' });
        if (query.trim()) params.set('q', query.trim());
        if (status !== 'all') params.set('reading_status', status);

        const response = await apiFetch<PaperListResponse>(`/papers?${params.toString()}`, {}, token);
        setPapers(response.items);
        setTotal(response.total);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load papers');
      } finally {
        setLoading(false);
      }
    })();
  }, [token, query, status, page]);

  return (
    <div className="stack-block">
      <div className="toolbar">
        <input
          value={query}
          onChange={(e) => {
            setPage(1);
            setQuery(e.target.value);
          }}
          className="toolbar-input"
          placeholder="Search title, author, keyword, DOI"
        />
        <select
          value={status}
          onChange={(e) => {
            setPage(1);
            setStatus(e.target.value);
          }}
          className="toolbar-select"
        >
          <option value="all">All statuses</option>
          <option value="unread">Unread</option>
          <option value="reading">Reading</option>
          <option value="read">Read</option>
        </select>
      </div>

      <div className="meta-row">
        <span>{total} results</span>
        <NavLink to="/papers/upload" className="text-link">+ Add paper</NavLink>
      </div>

      {error && <div className="error-box">{error}</div>}

      {loading ? (
        <div className="empty-state">Loading library…</div>
      ) : (
        <div className="paper-grid">
          {papers.length ? papers.map((paper) => <PaperCard key={paper.id} paper={paper} />) : <div className="empty-state">No papers match this filter.</div>}
        </div>
      )}

      <div className="pagination-row">
        <button type="button" className="secondary-button" disabled={page <= 1} onClick={() => setPage((current) => Math.max(current - 1, 1))}>
          Previous
        </button>
        <span>Page {page}</span>
        <button type="button" className="secondary-button" disabled={papers.length < 12} onClick={() => setPage((current) => current + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}

function UploadPage() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a PDF file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setError('');
      const response = await apiFetch<Paper>('/papers/upload', { method: 'POST', body: formData }, token);
      navigate(`/papers/${response.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to upload PDF');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-card narrow-panel">
      <h2>Upload a PDF</h2>
      <p className="muted">The backend will validate the file, extract text, chunk it, and begin indexing.</p>

      <form onSubmit={onSubmit} className="stack-form">
        {error && <div className="error-box">{error}</div>}
        <label>
          <span>PDF file</span>
          <input type="file" accept="application/pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0] ?? null)} />
        </label>

        <button type="submit" className="primary-button" disabled={loading || !file}>
          {loading ? 'Uploading...' : 'Upload paper'}
        </button>
      </form>
    </div>
  );
}

function PaperDetailPage() {
  const { paperId } = useParams();
  const { token } = useAuth();
  const navigate = useNavigate();
  const [paper, setPaper] = useState<Paper | null>(null);
  const [form, setForm] = useState({ title: '', authors: '', domain: '', notes: '', reading_status: 'unread' as Paper['reading_status'], tags: '' });
  const [saving, setSaving] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [summary, setSummary] = useState<{ content: string; citations: Array<Record<string, unknown>> } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [extractingAuthors, setExtractingAuthors] = useState(false);
  const [pdfDocument, setPdfDocument] = useState<{ paperId: string; url: string } | null>(null);
  const [pdfError, setPdfError] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!paperId || !token) return;

    void (async () => {
      try {
        const response = await apiFetch<Paper>(`/papers/${paperId}`, { method: 'GET' }, token);
        setPaper(response);
        setForm({
          title: response.title,
          authors: response.authors.join(', '),
          domain: response.domain ?? '',
          notes: response.notes ?? '',
          reading_status: response.reading_status,
          tags: (response.tags ?? []).join(', '),
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load paper');
      }
    })();
  }, [paperId, token]);

  useEffect(() => {
    if (!paper?.id || !paper.has_file || !token) return;
    let cancelled = false;
    let objectUrl: string | null = null;
    setPdfError('');

    void (async () => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/papers/${paper.id}/file`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'The PDF file is not available on the server.' : 'Unable to load the PDF.');
        }
        objectUrl = URL.createObjectURL(await response.blob());
        if (cancelled) {
          URL.revokeObjectURL(objectUrl);
        } else {
          setPdfDocument({ paperId: paper.id, url: objectUrl });
        }
      } catch (err) {
        if (!cancelled) setPdfError(err instanceof Error ? err.message : 'Unable to load the PDF.');
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [paper?.id, paper?.has_file, token]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!paperId || !paper) return;

    setSaving(true);
    setError('');

    try {
      const payload = {
        title: form.title,
        authors: form.authors.split(/[;,]/).map((value) => value.trim()).filter(Boolean),
        domain: form.domain.trim() || null,
        notes: form.notes,
        reading_status: form.reading_status,
        tags: form.tags.split(',').map((value) => value.trim()).filter(Boolean),
      };

      const updated = await apiFetch<Paper>(`/papers/${paperId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      }, token);

      setPaper(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to save paper');
    } finally {
      setSaving(false);
    }
  };

  const createSummary = async () => {
    if (!paperId || !paper || !token) return;
    setSummarizing(true);
    setError('');
    setSummary(null);
    try {
      const session = await apiFetch<ChatSession>('/chat/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: `Summary: ${paper.title.slice(0, 170)}`, paper_ids: [paperId] }),
      }, token);
      const response = await apiFetch<{ content: string; citations: Array<Record<string, unknown>> }>(
        `/chat/sessions/${session.id}/messages`,
        {
          method: 'POST',
          body: JSON.stringify({
            question: 'Summarize this paper. Organize the answer into: research problem and objective, methodology, data or datasets, key findings, limitations, and future work. Use only evidence from this paper and cite the relevant page for each section. If a section is not supported by the text, say so.',
          }),
        },
        token,
      );
      setSummary(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to summarize this paper');
    } finally {
      setSummarizing(false);
    }
  };

  const deletePaper = async () => {
    if (!paperId || !token || !window.confirm(`Permanently delete “${paper?.title ?? 'this paper'}” and its indexed content?`)) return;
    setDeleting(true);
    setError('');
    try {
      await apiFetch<void>(`/papers/${paperId}`, { method: 'DELETE' }, token);
      navigate('/papers');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete paper');
      setDeleting(false);
    }
  };

  const refreshAuthors = async () => {
    if (!paperId || !token) return;
    setExtractingAuthors(true);
    setError('');
    try {
      const updated = await apiFetch<Paper>(`/papers/${paperId}/refresh-metadata`, { method: 'POST' }, token);
      setPaper(updated);
      setForm((current) => ({ ...current, authors: updated.authors.join(', ') }));
      if (!updated.authors.length) {
        setError('No author names were detected. You can enter them in the metadata form.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to extract author names');
    } finally {
      setExtractingAuthors(false);
    }
  };

  if (!paper) {
    return <div className="empty-state">{error || 'Loading paper…'}</div>;
  }

  return (
    <div className="detail-layout">
      <div className="panel-card detail-panel">
        <div className="paper-header">
          <div>
            <p className="eyebrow">Paper</p>
            <h2>{paper.title}</h2>
          </div>
          <span className={`status-badge ${paper.reading_status}`}>{paper.reading_status}</span>
        </div>

        <div className="meta-grid">
          <div>
            <label>Authors</label>
            <p>{paper.authors.join(', ') || 'Not available'}</p>
            {!paper.authors.length && paper.has_file && (
              <button type="button" className="text-button" onClick={() => void refreshAuthors()} disabled={extractingAuthors}>
                {extractingAuthors ? 'Checking PDF…' : 'Try extracting authors'}
              </button>
            )}
          </div>
          <div><label>Publication</label><p>{paper.publication || 'Not available'}</p></div>
          <div><label>Year</label><p>{paper.year ?? '—'}</p></div>
          <div><label>DOI</label><p>{paper.doi || '—'}</p></div>
        </div>

        <div className="abstract-box">
          <h3>Abstract</h3>
          <p>{paper.abstract || 'No abstract extracted yet.'}</p>
        </div>

        {error && <div className="error-box">{error}</div>}
        <div className="topic-section">
          <strong>Research topic</strong>
          <div className="tag-row">
            {paper.domain && <span className="topic-chip">{paper.domain}</span>}
            {(paper.keywords ?? []).slice(0, 5).map((keyword) => <span key={keyword} className="topic-chip">{keyword}</span>)}
            {!paper.domain && !paper.keywords?.length && <span className="muted">Add a topic or keywords in metadata.</span>}
          </div>
        </div>
        <div className="tag-row">
          {(paper.tags ?? []).map((tag) => <span key={tag} className="tag">{tag}</span>)}
        </div>

        {paper.processing_error && <div className="warning-box">{paper.processing_error}</div>}
        <div className="paper-detail-actions">
          <button type="button" className="primary-button action-button" onClick={() => void createSummary()} disabled={summarizing || paper.processing_status !== 'completed'}>
            <Sparkles size={16} /> {summarizing ? 'Summarizing…' : 'Summarize paper'}
          </button>
          <button type="button" className="danger-button action-button" onClick={() => void deletePaper()} disabled={deleting}>
            <Trash2 size={16} /> {deleting ? 'Deleting…' : 'Delete paper'}
          </button>
        </div>
        {paper.processing_status !== 'completed' && <p className="muted">Summary requires a PDF with successfully extracted text.</p>}
        {summary && (
          <div className="summary-output">
            <h3>Paper summary</h3>
            <p>{summary.content}</p>
            {summary.citations.length > 0 && (
              <div className="citation-box">
                {summary.citations.map((citation, index) => (
                  <span key={index}>{String(citation.paper_title ?? paper.title)}{citation.page ? ` · p. ${String(citation.page)}` : ''}</span>
                ))}
              </div>
            )}
            <NavLink to="/chat" className="text-link">Continue in chat</NavLink>
          </div>
        )}
      </div>

      <div className="panel-card detail-panel">
        <h3>Edit metadata</h3>

        <form onSubmit={onSubmit} className="stack-form">
          <label>
            <span>Title</span>
            <input value={form.title} onChange={(e) => setForm((current) => ({ ...current, title: e.target.value }))} />
          </label>
          <label>
            <span>Authors (separate names with commas or semicolons)</span>
            <input value={form.authors} onChange={(e) => setForm((current) => ({ ...current, authors: e.target.value }))} placeholder="e.g. Alex Kim; Sam Patel" />
          </label>
          <label>
            <span>Primary research topic / domain</span>
            <input value={form.domain} onChange={(e) => setForm((current) => ({ ...current, domain: e.target.value }))} placeholder="e.g. Information retrieval" />
          </label>
          <label>
            <span>Reading status</span>
            <select value={form.reading_status} onChange={(e) => setForm((current) => ({ ...current, reading_status: e.target.value as Paper['reading_status'] }))}>
              <option value="unread">Unread</option>
              <option value="reading">Reading</option>
              <option value="read">Read</option>
            </select>
          </label>
          <label>
            <span>Tags</span>
            <input value={form.tags} onChange={(e) => setForm((current) => ({ ...current, tags: e.target.value }))} placeholder="e.g. NLP, RAG, evaluation" />
          </label>
          <label>
            <span>Notes</span>
            <textarea value={form.notes} onChange={(e) => setForm((current) => ({ ...current, notes: e.target.value }))} rows={6} />
          </label>
          <button type="submit" className="primary-button" disabled={saving}>
            {saving ? 'Saving...' : 'Save updates'}
          </button>
        </form>
      </div>
      {paper.has_file && (
        <section className="panel-card pdf-viewer-card">
          <div className="panel-header">
            <div>
              <h3>Full paper</h3>
              <p className="muted">View the uploaded PDF or open it in a separate tab.</p>
            </div>
            {pdfDocument?.paperId === paper.id && (
              <div className="export-actions">
                <a className="secondary-button pdf-link-button" href={pdfDocument.url} target="_blank" rel="noreferrer">Open in new tab</a>
                <a className="primary-button pdf-link-button" href={pdfDocument.url} download={`${paper.title.replace(/[^\w.-]+/g, '_')}.pdf`}>Download PDF</a>
              </div>
            )}
          </div>
          {pdfError && <div className="error-box">{pdfError}</div>}
          {pdfDocument?.paperId === paper.id ? (
            <iframe className="pdf-frame" src={pdfDocument.url} title={`PDF: ${paper.title}`} />
          ) : !pdfError ? <div className="empty-state">Loading PDF…</div> : null}
        </section>
      )}
    </div>
  );
}

function SearchPage() {
  const { token } = useAuth();
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'keyword' | 'semantic' | 'hybrid'>('hybrid');
  const [limit, setLimit] = useState(8);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!token || !query.trim()) return;

    try {
      setLoading(true);
      setError('');
      const response = await apiFetch<SearchResponse>(`/search?q=${encodeURIComponent(query.trim())}&mode=${mode}&limit=${limit}`, { method: 'GET' }, token);
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to search library');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack-block">
      <form onSubmit={handleSubmit} className="search-form">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all papers and chunks" />
        <select value={mode} onChange={(e) => setMode(e.target.value as 'keyword' | 'semantic' | 'hybrid')}>
          <option value="hybrid">Hybrid</option>
          <option value="keyword">Keyword</option>
          <option value="semantic">Semantic</option>
        </select>
        <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
          <option value={5}>5</option>
          <option value={8}>8</option>
          <option value={10}>10</option>
          <option value={15}>15</option>
        </select>
        <button type="submit" className="primary-button" disabled={loading || !query.trim()}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="error-box">{error}</div>}

      {results && (
        <div className="panel-card">
          <div className="panel-header">
            <h2>Results for “{results.query}”</h2>
            <span className="status-badge neutral">{results.mode}</span>
          </div>

          {results.results.length ? (
            <div className="search-result-list">
              {results.results.map((result) => (
                <div key={`${result.paper_id}-${result.page ?? 'all'}`} className="search-result-item">
                  <div className="result-head">
                    <strong>{result.title}</strong>
                    <span>{result.score.toFixed(3)}</span>
                  </div>
                  <p>{result.snippet}</p>
                  <div className="result-meta">
                    <span>{result.authors.join(', ') || 'Unknown author'}</span>
                    <span>{result.matched_by.join(', ') || 'keyword'}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">No hits found.</div>
          )}
        </div>
      )}
    </div>
  );
}

function ChatPage() {
  const { token } = useAuth();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [session, setSession] = useState<ChatSessionDetail | null>(null);
  const [paperIds, setPaperIds] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const loadSessions = async () => {
    if (!token) return;
    const response = await apiFetch<ChatSession[]>('/chat/sessions', { method: 'GET' }, token);
    setSessions(response);
    if (!selectedSessionId && response[0]) {
      setSelectedSessionId(response[0].id);
    }
  };

  useEffect(() => {
    if (!token) return;

    void (async () => {
      try {
        const paperResponse = await apiFetch<PaperListResponse>('/papers?page=1&page_size=50', {}, token);
        setPapers(paperResponse.items);
        await loadSessions();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load chat data');
      }
    })();
  }, [token]);

  useEffect(() => {
    if (!token || !selectedSessionId) return;

    void (async () => {
      try {
        const response = await apiFetch<ChatSessionDetail>(`/chat/sessions/${selectedSessionId}`, { method: 'GET' }, token);
        setSession(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load chat session');
      }
    })();
  }, [selectedSessionId, token]);

  const createSession = async () => {
    if (!token) return;
    try {
      const response = await apiFetch<ChatSession>('/chat/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: title || 'New chat', paper_ids: paperIds }),
      }, token);

      setTitle('');
      setPaperIds([]);
      setSelectedSessionId(response.id);
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create a chat session');
    }
  };

  const sendMessage = async () => {
    if (!token || !selectedSessionId || !question.trim()) return;

    try {
      setLoading(true);
      setError('');
      await apiFetch('/chat/sessions/' + selectedSessionId + '/messages',
        {
          method: 'POST',
          body: JSON.stringify({ question: question.trim() }),
        },
        token,
      );

      setQuestion('');
      const refreshed = await apiFetch<ChatSessionDetail>(`/chat/sessions/${selectedSessionId}`, { method: 'GET' }, token);
      setSession(refreshed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to send message');
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async (sessionId: string) => {
    if (!token || !window.confirm('Delete this conversation and its messages?')) return;
    setError('');
    try {
      await apiFetch<void>(`/chat/sessions/${sessionId}`, { method: 'DELETE' }, token);
      const remaining = await apiFetch<ChatSession[]>('/chat/sessions', { method: 'GET' }, token);
      setSessions(remaining);
      if (selectedSessionId === sessionId) {
        setSelectedSessionId(remaining[0]?.id ?? null);
        setSession(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to delete conversation');
    }
  };

  const copyAnswer = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      window.setTimeout(() => setCopiedMessageId(null), 1800);
    } catch {
      setError('Could not copy the answer. Check clipboard permissions and try again.');
    }
  };

  return (
    <div className="chat-layout">
      <aside className="panel-card chat-sidebar">
        <h3>Sessions</h3>

        <div className="stack-form slim-form">
          <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Chat title" />
          <div className="paper-picker">
            {papers.map((paper) => (
              <label key={paper.id} className="checkbox-row">
                <input
                  type="checkbox"
                  checked={paperIds.includes(paper.id)}
                  onChange={(e) => {
                    setPaperIds((current) => e.target.checked ? [...current, paper.id] : current.filter((id) => id !== paper.id));
                  }}
                />
                <span>{paper.title}</span>
              </label>
            ))}
          </div>
          <button type="button" className="secondary-button" onClick={createSession}>New session</button>
        </div>

        <div className="session-list">
          {sessions.map((sessionItem) => (
            <div className="session-row" key={sessionItem.id}>
              <button
                type="button"
                className={selectedSessionId === sessionItem.id ? 'session-item active' : 'session-item'}
                onClick={() => setSelectedSessionId(sessionItem.id)}
              >
                <strong>{sessionItem.title}</strong>
                <small>{sessionItem.paper_ids.length} paper scopes</small>
              </button>
              <button type="button" className="session-delete" onClick={() => void deleteSession(sessionItem.id)} aria-label={`Delete ${sessionItem.title}`}>
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="panel-card chat-panel">
        {error && <div className="error-box">{error}</div>}

        {session ? (
          <>
            <div className="chat-thread">
              {session.messages.length ? session.messages.map((message) => (
                <div key={message.id} className={message.role === 'assistant' ? 'chat-bubble assistant' : 'chat-bubble user'}>
                  <div className="bubble-label">{message.role === 'assistant' ? 'Assistant' : 'You'}</div>
                  {message.role === 'assistant' && (
                    <small className="message-mode">
                      {message.model ?? (message.mode === 'extractive' ? 'Extractive answer · no LLM configured' : 'AI answer')}
                    </small>
                  )}
                  <p>{message.content}</p>
                  {message.citations && message.citations.length > 0 && (
                    <div className="citation-box">
                      {message.citations.map((citation, index) => (
                        <span key={`${message.id}-citation-${index}`}>
                          {String(citation.paper_title ?? 'Citation')}{citation.page ? ` · p. ${String(citation.page)}` : ''}
                        </span>
                      ))}
                    </div>
                  )}
                  {message.warnings?.map((warning, index) => <small className="message-warning" key={index}>{warning}</small>)}
                  {message.role === 'assistant' && (
                    <button type="button" className="copy-answer-button" onClick={() => void copyAnswer(message.id, message.content)}>
                      <Copy size={14} /> {copiedMessageId === message.id ? 'Copied' : 'Copy answer'}
                    </button>
                  )}
                </div>
              )) : <div className="empty-state">No messages yet. Ask your first question.</div>}
            </div>

            <div className="chat-compose">
              <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={4} placeholder="Ask about this paper set..." />
              <button type="button" className="primary-button" onClick={sendMessage} disabled={loading || !question.trim()}>
                {loading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </>
        ) : (
          <div className="empty-state">Select a session or create a new one.</div>
        )}
      </section>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent: 'blue' | 'amber' | 'green' | 'purple' }) {
  return (
    <div className={`stat-card ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PaperSummaryRow({ paper }: { paper: Paper }) {
  return (
    <div className="summary-row">
      <div>
        <NavLink to={`/papers/${paper.id}`} className="link-strong">{paper.title}</NavLink>
        <p>{paper.authors.join(', ') || 'Unknown author'} · {paper.year ?? 'Year unknown'}</p>
      </div>
      <div className="summary-row-side">
        <span className={`status-badge ${paper.reading_status}`}>{paper.reading_status}</span>
        <small>{formatDistanceToNow(new Date(paper.updated_at), { addSuffix: true })}</small>
      </div>
    </div>
  );
}

function PaperCard({ paper }: { paper: Paper }) {
  const topics = [paper.domain, ...(paper.keywords ?? [])].filter((value): value is string => Boolean(value)).slice(0, 4);
  return (
    <article className="paper-card">
      <div className="paper-card-header">
        <div>
          <NavLink to={`/papers/${paper.id}`} className="paper-title">{paper.title}</NavLink>
          <p>{paper.authors.join(', ') || 'Unknown author'}</p>
        </div>
        <span className={`status-badge ${paper.reading_status}`}>{paper.reading_status}</span>
      </div>

      <p className="paper-abstract">{paper.abstract ? paper.abstract.slice(0, 180) + (paper.abstract.length > 180 ? '…' : '') : 'No abstract preview yet.'}</p>

      <div className="paper-card-meta">
        <span>{paper.year ?? 'Year unspecified'}</span>
        <span>{paper.publication || 'Unspecified venue'}</span>
      </div>

      <div className="tag-row compact">
        {(paper.tags ?? []).slice(0, 3).map((tag) => (
          <span key={tag} className="tag">{tag}</span>
        ))}
      </div>

      <div className="paper-card-footer">
        <span>{paper.processing_status}</span>
        <span>{format(new Date(paper.updated_at), 'MMM d, yyyy')}</span>
      </div>
      <div className="paper-topic-footer">
        <strong>Topic</strong>
        {topics.length ? topics.map((topic) => <span className="topic-chip" key={topic}>{topic}</span>) : <span className="muted">Not identified</span>}
      </div>
    </article>
  );
}

export default App;
