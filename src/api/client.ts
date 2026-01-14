export class ApiError extends Error {
  status: number;
  statusText: string;
  data?: unknown;

  constructor(message: string, options: { status: number; statusText: string; data?: unknown }) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status;
    this.statusText = options.statusText;
    this.data = options.data;
  }
}

export interface RequestOptions extends RequestInit {
  parseErrorResponse?: boolean;
}

const parseErrorPayload = async (response: Response) => {
  const headerGetter =
    typeof response.headers?.get === 'function'
      ? response.headers.get.bind(response.headers)
      : null;
  const contentType = headerGetter ? headerGetter('content-type') || '' : '';
  try {
    if (contentType.includes('application/json')) {
      return await response.clone().json();
    }
    if (contentType.includes('text/')) {
      return await response.clone().text();
    }
  } catch (error) {
    console.warn('Failed to parse API error payload', error);
  }
  return undefined;
};

export const request = async (
  input: RequestInfo | URL,
  init: RequestOptions = {},
): Promise<Response> => {
  const headers = new Headers(init.headers as HeadersInit | undefined);
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }

  const response = await fetch(input, { ...init, headers });

  if (!response.ok) {
    const data = init.parseErrorResponse === false ? undefined : await parseErrorPayload(response);
    throw new ApiError(`Request failed with status ${response.status}`, {
      status: response.status,
      statusText: response.statusText,
      data,
    });
  }

  return response;
};

export const getJson = async <T>(input: RequestInfo | URL, init?: RequestOptions): Promise<T> => {
  const response = await request(input, init);
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
};

export const postJson = async <TResponse, TBody = unknown>(
  input: RequestInfo | URL,
  body?: TBody,
  init: RequestOptions = {},
): Promise<TResponse> => {
  const headers = new Headers(init.headers as HeadersInit | undefined);
  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await request(input, {
    ...init,
    method: init.method ?? 'POST',
    headers,
    body: body === undefined ? init.body : JSON.stringify(body),
  });

  if (response.status === 204) {
    return undefined as TResponse;
  }
  return response.json() as Promise<TResponse>;
};

export const deleteJson = async <T>(
  input: RequestInfo | URL,
  init: RequestOptions = {},
): Promise<T> => {
  return postJson<T>(input, undefined, { ...init, method: 'DELETE' });
};

export const apiClient = {
  request,
  getJson,
  postJson,
  deleteJson,
};

export type ApiClient = typeof apiClient;
