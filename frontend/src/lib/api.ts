import type { AuthTokens, User } from '../types';

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

export function getApiBaseUrl() {
  return API_BASE_URL;
}

export async function apiFetch<T>(path: string, options: RequestInit = {}, token?: string | null): Promise<T> {
  const headers = new Headers(options.headers ?? {});

  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  let payload: unknown = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const detail = typeof payload === 'object' && payload !== null && 'detail' in payload
      ? (payload as { detail?: string }).detail
      : null;

    throw new Error(detail ?? 'Request failed');
  }

  return payload as T;
}

export async function loginRequest(email: string, password: string): Promise<AuthTokens> {
  return apiFetch<AuthTokens>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

export async function registerRequest(fullName: string, email: string, password: string): Promise<AuthTokens> {
  return apiFetch<AuthTokens>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ full_name: fullName, email, password }),
  });
}

export async function fetchCurrentUser(token: string): Promise<User> {
  return apiFetch<User>('/auth/me', { method: 'GET' }, token);
}
