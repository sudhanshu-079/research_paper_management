export type PaperReadingStatus = 'unread' | 'reading' | 'read';

export type User = {
  id: string;
  email: string;
  full_name: string;
};

export type AuthTokens = {
  access_token: string;
  refresh_token: string;
};

export type Paper = {
  id: string;
  title: string;
  authors: string[];
  abstract?: string | null;
  publication?: string | null;
  year?: number | null;
  doi?: string | null;
  keywords: string[];
  tags: string[];
  domain?: string | null;
  notes?: string | null;
  reading_status: PaperReadingStatus;
  processing_status: string;
  needs_review: boolean;
  has_file: boolean;
  processing_error?: string | null;
  index_status?: string | null;
  embedding_model?: string | null;
  created_at: string;
  updated_at: string;
};

export type PaperListResponse = {
  items: Paper[];
  total: number;
  page: number;
  page_size: number;
};

export type ChatSession = {
  id: string;
  title: string;
  paper_ids: string[];
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  supported?: boolean | null;
  citations?: Array<Record<string, unknown>>;
  warnings?: string[];
  mode?: string | null;
  model?: string | null;
  latency_ms?: number | null;
  created_at: string;
};

export type ChatSessionDetail = ChatSession & {
  messages: ChatMessage[];
};

export type SearchHit = {
  paper_id: string;
  title: string;
  year?: number | null;
  authors: string[];
  score: number;
  keyword_score?: number | null;
  semantic_score?: number | null;
  matched_by: string[];
  page?: number | null;
  snippet: string;
};

export type SearchResponse = {
  query: string;
  mode: 'keyword' | 'semantic' | 'hybrid';
  results: SearchHit[];
};
