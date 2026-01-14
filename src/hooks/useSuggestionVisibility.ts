/**
 * Global Suggestion Visibility Hook
 * Manages timer cancellation and user interaction events for suggestions
 */

import { useEffect, useCallback } from 'react';

export interface SuggestionVisibilityManager {
  cancelAllTimers: () => void;
  pauseTimers: () => void;
  resumeTimers: () => void;
}

let isPaused = false;

export const useSuggestionVisibility = (): SuggestionVisibilityManager => {
  const cancelAllTimers = useCallback(() => {
    // Dispatch custom event to cancel all timers
    window.dispatchEvent(new CustomEvent('suggestion-cancel-timers'));
  }, []);

  const pauseTimers = useCallback(() => {
    isPaused = true;
    window.dispatchEvent(new CustomEvent('suggestion-pause-timers'));
  }, []);

  const resumeTimers = useCallback(() => {
    isPaused = false;
    window.dispatchEvent(new CustomEvent('suggestion-resume-timers'));
  }, []);

  // Set up global event listeners for user interactions
  useEffect(() => {
    let scrollTimeout: number | null = null;
    let isScrolling = false;

    const handleScroll = () => {
      if (!isScrolling) {
        isScrolling = true;
        pauseTimers();
      }

      // Clear existing timeout
      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }

      // Set new timeout to resume timers after scrolling stops
      scrollTimeout = window.setTimeout(() => {
        isScrolling = false;
        resumeTimers();
      }, 150); // Resume 150ms after scroll stops
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      // Cancel timers if user starts typing anywhere
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        cancelAllTimers();
      }
    };

    const handleFocus = (e: FocusEvent) => {
      // Cancel timers if user focuses on input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        cancelAllTimers();
      }
    };

    const handleVisibilityChange = () => {
      // Pause timers when tab is not visible
      if (document.hidden) {
        pauseTimers();
      } else {
        resumeTimers();
      }
    };

    const handleUserActivity = () => {
      // Cancel timers on any significant user interaction
      cancelAllTimers();
    };

    // Add event listeners
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('focus', handleFocus, true);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Optional: Cancel on any click outside suggestion areas
    window.addEventListener('click', handleUserActivity);

    return () => {
      // Cleanup
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('focus', handleFocus, true);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('click', handleUserActivity);

      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }
    };
  }, [cancelAllTimers, pauseTimers, resumeTimers]);

  return {
    cancelAllTimers,
    pauseTimers,
    resumeTimers,
  };
};

export default useSuggestionVisibility;
