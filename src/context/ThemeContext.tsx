import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { StorageKeys } from '@/constants/storage';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { applyThemeToDocument, getSystemTheme, type ThemeMode } from '@/utils/theme';

type Theme = ThemeMode;

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useLocalStorage<Theme>(StorageKeys.theme, () => getSystemTheme());

  // Update document class, data-theme attribute and localStorage when theme changes
  useEffect(() => {
    applyThemeToDocument(theme);
  }, [theme]);

  // Toggle between light and dark
  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  return <ThemeContext.Provider value={{ theme, toggleTheme }}>{children}</ThemeContext.Provider>;
};

// Custom hook for accessing theme
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
