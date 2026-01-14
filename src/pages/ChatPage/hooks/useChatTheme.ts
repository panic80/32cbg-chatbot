import { useEffect } from 'react';
import { applyThemeToDocument, normalizeTheme } from '@/utils/theme';

export const useChatTheme = (theme: string, propTheme?: string) => {
  // Apply theme changes to document only if not managed by parent
  useEffect(() => {
    if (!propTheme) {
      const normalized = normalizeTheme(theme);
      applyThemeToDocument(normalized);
    }
  }, [theme, propTheme]);
};
