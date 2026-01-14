import { apiClient, ApiError } from './client';

export type VisitSummary = {
  totalVisits: number;
  firstVisit: string | null;
  lastVisit: string | null;
  dailyCounts: Array<{ date: string; count: number }>;
};

export type VisitSummaryFilters = {
  startAt?: string;
  endAt?: string;
  path?: string;
};

type VisitEventPayload = {
  path: string;
  referrer?: string | null;
  sessionId?: string | null;
  locale?: string | null;
  title?: string | null;
  viewport?: string | null;
  metadata?: Record<string, unknown>;
};

export async function sendVisitEvent(event: VisitEventPayload): Promise<boolean> {
  try {
    await apiClient.request('/api/analytics/visit', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(event),
      keepalive: true,
      parseErrorResponse: false,
    });
    return true;
  } catch (error) {
    if (error instanceof ApiError) {
      if (error.status !== 503) {
        console.warn('Visit event call failed', error.status, error.statusText);
      }
      return false;
    }
    console.warn('Unable to record visit event', error);
    return false;
  }
}

export async function fetchVisitSummary(filters: VisitSummaryFilters = {}): Promise<VisitSummary> {
  const params = new URLSearchParams();

  if (filters.startAt) {
    params.set('startAt', filters.startAt);
  }
  if (filters.endAt) {
    params.set('endAt', filters.endAt);
  }
  if (filters.path) {
    params.set('path', filters.path);
  }

  const query = params.toString();
  const url =
    query.length > 0 ? `/api/admin/analytics/visits?${query}` : '/api/admin/analytics/visits';

  try {
    const data = await apiClient.getJson<{ data?: VisitSummary }>(url, {
      parseErrorResponse: true,
    });

    return (
      data?.data ?? {
        totalVisits: 0,
        firstVisit: null,
        lastVisit: null,
        dailyCounts: [],
      }
    );
  } catch (error) {
    if (error instanceof ApiError) {
      const errorData = error.data as Record<string, unknown> | null;
      const message =
        typeof errorData?.message === 'string'
          ? errorData.message
          : `Failed to load visit analytics (${error.status})`;
      throw new Error(message);
    }

    throw error;
  }
}
