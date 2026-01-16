import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { DEFAULT_CHAT_INPUT_HEIGHT_PX } from '@/constants';

interface UseChatInputStateReturn {
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  inputHeight: number;
}

/**
 * Manages chat input text state and dynamically measures input area height.
 * Also handles initial query parameter from URL (?q=...).
 */
export const useChatInputState = (): UseChatInputStateReturn => {
  const [input, setInput] = useState('');
  const [inputHeight, setInputHeight] = useState<number>(DEFAULT_CHAT_INPUT_HEIGHT_PX);
  const location = useLocation();

  // Handle initial query from URL
  useEffect(() => {
    try {
      const params = new URLSearchParams(location.search);
      const q = params.get('q');
      if (q && q.trim().length > 0) {
        setInput(q.trim());
      }
    } catch {
      // no-op
    }
  }, [location.search]);

  // Measure input element height dynamically
  useEffect(() => {
    const el = document.querySelector('[data-chat-input]') as HTMLElement | null;
    if (!el) return;

    const measure = () => {
      setInputHeight(el.getBoundingClientRect().height || DEFAULT_CHAT_INPUT_HEIGHT_PX);
    };
    measure();

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => measure());
      ro.observe(el);
    }

    window.addEventListener('resize', measure);
    return () => {
      ro?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  return {
    input,
    setInput,
    inputHeight,
  };
};
