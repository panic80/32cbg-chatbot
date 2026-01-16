import { useState, useCallback, useEffect } from 'react';
import { MENU_HIGHLIGHT_DURATION_MS } from '@/constants';

type MenuHighlight = 'none' | 'model' | 'short';

interface UseMenuHighlightReturn {
  menuOpen: boolean;
  setMenuOpen: (open: boolean) => void;
  menuHighlight: MenuHighlight;
  triggerMenu: (highlight: 'model' | 'short') => void;
  handleModePillClick: () => void;
  handleShortAnswerPillClick: () => void;
}

/**
 * Manages hamburger menu state and highlight animations for mode pills.
 */
export const useMenuHighlight = (): UseMenuHighlightReturn => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuHighlight, setMenuHighlight] = useState<MenuHighlight>('none');

  const triggerMenu = useCallback((highlight: 'model' | 'short') => {
    setMenuHighlight(highlight);
    setMenuOpen(true);
  }, []);

  const handleModePillClick = useCallback(() => triggerMenu('model'), [triggerMenu]);
  const handleShortAnswerPillClick = useCallback(() => triggerMenu('short'), [triggerMenu]);

  // Reset highlight when menu closes
  useEffect(() => {
    if (!menuOpen && menuHighlight !== 'none') {
      setMenuHighlight('none');
    }
  }, [menuOpen, menuHighlight]);

  // Auto-clear highlight after duration
  useEffect(() => {
    if (menuHighlight !== 'none') {
      const timer = setTimeout(() => setMenuHighlight('none'), MENU_HIGHLIGHT_DURATION_MS);
      return () => clearTimeout(timer);
    }
  }, [menuHighlight]);

  return {
    menuOpen,
    setMenuOpen,
    menuHighlight,
    triggerMenu,
    handleModePillClick,
    handleShortAnswerPillClick,
  };
};
