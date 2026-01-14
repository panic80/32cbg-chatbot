/**
 * Shared chat utility helpers used across the client application.
 * Consolidates previously split JS/TS implementations into a single typed module
 * so we avoid duplicate maintenance and subtle behavioural drift.
 */

export interface ChatSource {
  id: string;
  text: string;
  reference?: string;
}

export interface ParsedChatResponse {
  text: string;
  sources: ChatSource[];
  isFormatted: boolean;
}

/**
 * Generate a reasonably unique message identifier.
 */
export const generateMessageId = (): string => {
  const randomSegment = Math.random().toString(36).slice(2, 9);
  return `msg-${Date.now()}-${randomSegment}`;
};

/**
 * Lightweight text normalizer used before rendering markdown output.
 */
export const formatText = (text: string): string => {
  if (!text) return '';
  return text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => line.trim())
    .join('\n');
};

/**
 * Parse structured API responses into a reusable shape for the UI.
 */
export const parseApiResponse = (text: string, isSimplified = false): ParsedChatResponse => {
  if (!text) {
    throw new Error('Invalid response format from API');
  }

  const sections = text.split('\n').filter((line) => line.trim());

  const reference = sections
    .find((line) => line.startsWith('Reference:'))
    ?.replace('Reference:', '')
    .trim();
  const quote = sections
    .find((line) => line.startsWith('Quote:'))
    ?.replace('Quote:', '')
    .trim();
  const answer = sections
    .find((line) => line.startsWith('Answer:'))
    ?.replace('Answer:', '')
    .trim();
  const reason = sections
    .find((line) => line.startsWith('Reason:'))
    ?.replace('Reason:', '')
    .trim();

  if (answer) {
    let formattedText = answer;

    if (!isSimplified && reason) {
      formattedText += `\n\n**Detailed Explanation:**\n\n${reason}`;
    }

    const sources: ChatSource[] = [];

    if (quote && reference) {
      sources.push({
        id: reference.replace(/\s+/g, '-').toLowerCase() || 'source-reference',
        text: quote,
        reference,
      });
    }

    return {
      text: formattedText,
      sources,
      isFormatted: true,
    };
  }

  const sourceMatches = text.match(/\(Source \d+(?:, Source \d+)*\)/g) || [];
  const sources: ChatSource[] = [];

  if (sourceMatches.length > 0) {
    const allSources = new Set<number>();
    sourceMatches.forEach((match) => {
      const numbers = match.match(/\d+/g) || [];
      numbers.forEach((num) => allSources.add(Number.parseInt(num, 10)));
    });

    const sortedSources = Array.from(allSources).sort((a, b) => a - b);
    const referenceText =
      sortedSources.length === 1
        ? `Source ${sortedSources[0]}`
        : `Sources ${sortedSources.join(', ')}`;

    sources.push({
      id: `source-${sortedSources.join('-')}` || 'source-aggregated',
      text: 'Information referenced from the provided documentation',
      reference: referenceText,
    });
  }

  return {
    text,
    sources,
    isFormatted: true,
  };
};

/**
 * Format a date string into human-friendly chat separators.
 */
export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return 'Recent';
  }

  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return 'Today';
  }
  if (date.toDateString() === yesterday.toDateString()) {
    return 'Yesterday';
  }

  return date.toLocaleDateString(undefined, {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
  });
};

/**
 * Format timestamps for bubble footers.
 */
export const formatMessageTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
};

/** Determine if viewport should be considered mobile. */
export const isMobileDevice = (): boolean => window.innerWidth <= 768;

/** Detect whether the user has scrolled away from the bottom of a container. */
export const hasScrolledUp = (container: HTMLElement | null, threshold = 20): boolean => {
  if (!container) return false;
  const { scrollTop, scrollHeight, clientHeight } = container;
  return scrollHeight - scrollTop - clientHeight > threshold;
};

/** Detect whether the user is near the bottom of the scroll container. */
export const isScrolledToBottom = (container: HTMLElement | null, threshold = 20): boolean => {
  if (!container) return false;
  const { scrollTop, scrollHeight, clientHeight } = container;
  return scrollHeight - scrollTop - clientHeight <= threshold;
};

/** Scroll to the bottom of a container. */
export const scrollToBottom = (container: HTMLElement | null): void => {
  if (!container) return;
  container.scrollTop = container.scrollHeight;
};

/** Debounce helper to limit function invocation rate. */
export const debounce = <F extends (...args: unknown[]) => void>(
  func: F,
  waitFor: number,
): ((...args: Parameters<F>) => void) => {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<F>): void => {
    if (timeout !== null) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => func(...args), waitFor);
  };
};
