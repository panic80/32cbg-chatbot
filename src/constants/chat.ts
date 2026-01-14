export const SCROLL_BOTTOM_THRESHOLD_PX = 20;

export const DEFAULT_CHAT_INPUT_HEIGHT_PX = 96;
export const NEW_REPLY_PILL_MARGIN_PX = 12;

export const CHAT_TRUNCATE_CHARS = 400;
export const SOURCE_PREVIEW_COUNT = 3;

export const MAX_SUGGESTIONS = 3;
export const SUGGESTION_DIVERSITY_WEIGHT = 0.3;

export const FOLLOW_UP_COUNT = 3;
export const FALLBACK_FOLLOW_UP_COUNT = 2;

export const CHAT_PAGE_SIZE = 50;

export const SIMULATED_REGENERATE_DELAY_MS = 1500;

export const HISTORY_WINDOW_BY_MODEL: Record<string, number> = {
  'gpt-5-mini': 4,
  default: 10,
};
