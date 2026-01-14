import { useEffect, useState, useCallback } from 'react';

interface MobileKeyboardState {
  isKeyboardVisible: boolean;
  keyboardHeight: number;
}

/**
 * Hook to handle mobile virtual keyboard interactions
 * Provides keyboard visibility state and height for layout adjustments
 */
export const useMobileKeyboard = () => {
  const [keyboardState, setKeyboardState] = useState<MobileKeyboardState>({
    isKeyboardVisible: false,
    keyboardHeight: 0,
  });

  // Detect if the device is mobile
  const isMobile =
    /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
    (window.matchMedia && window.matchMedia('(max-width: 768px)').matches);

  useEffect(() => {
    if (!isMobile) return;

    let lastHeight = window.innerHeight;
    let keyboardTimer: ReturnType<typeof setTimeout>;

    const handleViewportChange = () => {
      const currentHeight = window.innerHeight;
      const heightDifference = lastHeight - currentHeight;

      // Clear any existing timer
      clearTimeout(keyboardTimer);

      // Keyboard is likely visible if viewport shrinks by more than 50px
      if (heightDifference > 50) {
        setKeyboardState({
          isKeyboardVisible: true,
          keyboardHeight: heightDifference,
        });

        // Update CSS custom property for keyboard height
        document.documentElement.style.setProperty('--keyboard-height', `${heightDifference}px`);
        document.body.classList.add('keyboard-visible');
        document.body.dataset.keyboardHeight = String(heightDifference);
      } else if (heightDifference < -50) {
        // Keyboard is hiding
        keyboardTimer = setTimeout(() => {
          setKeyboardState({
            isKeyboardVisible: false,
            keyboardHeight: 0,
          });
          document.documentElement.style.setProperty('--keyboard-height', '0px');
          document.body.classList.remove('keyboard-visible');
          delete document.body.dataset.keyboardHeight;
        }, 100);
      }

      lastHeight = currentHeight;
    };

    // Visual Viewport API (more reliable on modern browsers)
    if ('visualViewport' in window && window.visualViewport) {
      const viewport = window.visualViewport;

      const handleViewportUpdate = () => {
        const keyboardHeight = Math.max(
          0,
          window.innerHeight - viewport.height - viewport.offsetTop,
        );
        const isVisible = keyboardHeight > 20; // Lower threshold for better detection

        setKeyboardState({
          isKeyboardVisible: isVisible,
          keyboardHeight: keyboardHeight,
        });

        document.documentElement.style.setProperty('--keyboard-height', `${keyboardHeight}px`);

        // Also update body class for CSS hooks
        if (isVisible) {
          document.body.classList.add('keyboard-visible');
          document.body.dataset.keyboardHeight = String(keyboardHeight);
        } else {
          document.body.classList.remove('keyboard-visible');
          delete document.body.dataset.keyboardHeight;
        }
      };

      viewport.addEventListener('resize', handleViewportUpdate);
      viewport.addEventListener('scroll', handleViewportUpdate);

      return () => {
        viewport.removeEventListener('resize', handleViewportUpdate);
        viewport.removeEventListener('scroll', handleViewportUpdate);
        clearTimeout(keyboardTimer);
      };
    } else {
      // Fallback for older browsers
      window.addEventListener('resize', handleViewportChange);

      // Also listen for focus/blur on inputs as additional signals
      const handleFocus = () => {
        setTimeout(handleViewportChange, 300);
      };

      const handleBlur = () => {
        setTimeout(() => {
          setKeyboardState({
            isKeyboardVisible: false,
            keyboardHeight: 0,
          });
          document.documentElement.style.setProperty('--keyboard-height', '0px');
          document.body.classList.remove('keyboard-visible');
          delete document.body.dataset.keyboardHeight;
        }, 300);
      };

      document.addEventListener('focusin', handleFocus);
      document.addEventListener('focusout', handleBlur);

      return () => {
        window.removeEventListener('resize', handleViewportChange);
        document.removeEventListener('focusin', handleFocus);
        document.removeEventListener('focusout', handleBlur);
        clearTimeout(keyboardTimer);
      };
    }
  }, [isMobile]);

  // Scroll input into view when keyboard opens
  const scrollInputIntoView = useCallback(
    (element: HTMLElement | null) => {
      if (!element || !keyboardState.isKeyboardVisible) return;

      setTimeout(() => {
        // For iOS, we need to handle this differently
        if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
          // Use a more conservative approach for iOS
          element.scrollIntoView({
            behavior: 'smooth',
            block: 'end',
            inline: 'nearest',
          });
        } else {
          element.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
            inline: 'nearest',
          });
        }
      }, 300);
    },
    [keyboardState.isKeyboardVisible],
  );

  // Force close keyboard
  const closeKeyboard = useCallback(() => {
    const activeElement = document.activeElement as HTMLElement;
    if (activeElement && 'blur' in activeElement) {
      activeElement.blur();
    }
  }, []);

  return {
    isKeyboardVisible: keyboardState.isKeyboardVisible,
    keyboardHeight: keyboardState.keyboardHeight,
    scrollInputIntoView,
    closeKeyboard,
    isMobile,
  };
};

// Hook for detecting scroll direction
export const useScrollDirection = () => {
  const [scrollDirection, setScrollDirection] = useState<'up' | 'down' | null>(null);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    let ticking = false;

    const updateScrollDirection = () => {
      const scrollY = window.pageYOffset;

      if (Math.abs(scrollY - lastScrollY) < 5) {
        ticking = false;
        return;
      }

      setScrollDirection(scrollY > lastScrollY ? 'down' : 'up');
      setLastScrollY(scrollY);
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(updateScrollDirection);
        ticking = true;
      }
    };

    window.addEventListener('scroll', onScroll);

    return () => window.removeEventListener('scroll', onScroll);
  }, [lastScrollY]);

  return scrollDirection;
};
