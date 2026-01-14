export type ThemeMode = 'light' | 'dark';

export const getSystemTheme = (): ThemeMode => {
  if (typeof window === 'undefined') {
    return 'light';
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

export const normalizeTheme = (value: string | null | undefined): ThemeMode => {
  if (!value) {
    return 'light';
  }

  const trimmed = value.replace(/^"|"$/g, '');
  return trimmed === 'dark' ? 'dark' : 'light';
};

export const applyThemeToDocument = (theme: ThemeMode): void => {
  if (typeof document === 'undefined') {
    return;
  }

  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  root.classList.add(theme);
  root.classList.toggle('dark', theme === 'dark');
  root.setAttribute('data-theme', theme);
};
