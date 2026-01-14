import { useEffect, useState } from 'react';

const getIsMobile = (breakpoint: number) => {
  if (typeof window === 'undefined') return false;
  return window.innerWidth <= breakpoint;
};

const updateRootAttribute = (isMobile: boolean) => {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-mobile', isMobile ? 'true' : 'false');
};

export const useMobileFlag = (breakpoint = 768) => {
  const [isMobile, setIsMobile] = useState(() => getIsMobile(breakpoint));

  useEffect(() => {
    updateRootAttribute(isMobile);
  }, [isMobile]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    let resizeTimeout: number | null = null;

    const commitUpdate = () => {
      const nextIsMobile = getIsMobile(breakpoint);
      setIsMobile((prev) => (prev === nextIsMobile ? prev : nextIsMobile));
    };

    const handleResize = () => {
      if (resizeTimeout) {
        clearTimeout(resizeTimeout);
      }
      resizeTimeout = window.setTimeout(commitUpdate, 150);
    };

    commitUpdate();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeTimeout) {
        clearTimeout(resizeTimeout);
      }
    };
  }, [breakpoint]);

  return isMobile;
};

export default useMobileFlag;
