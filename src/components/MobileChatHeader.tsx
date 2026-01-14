import React from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, Settings, Menu } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface MobileChatHeaderProps {
  title: string;
  subtitle?: string;
  isVisible: boolean;
  onBack?: () => void;
  onSettings?: () => void;
  onMenu?: () => void;
  className?: string;
}

/**
 * Mobile-optimized collapsible chat header
 * Hides on scroll down, shows on scroll up
 */
export const MobileChatHeader: React.FC<MobileChatHeaderProps> = ({
  title,
  subtitle,
  isVisible,
  onBack,
  onSettings,
  onMenu,
  className = '',
}) => {
  return (
    <motion.header
      className={cn(
        'fixed top-0 left-0 right-0 z-40',
        'bg-background/95 backdrop-blur-lg border-b border-border',
        'transition-transform duration-300 ease-in-out',
        className,
      )}
      initial={{ y: 0 }}
      animate={{ y: isVisible ? 0 : -100 }}
      style={{
        paddingTop: 'env(safe-area-inset-top, 0)',
      }}
    >
      <div className="h-14 px-4 flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center gap-2">
          {onBack && (
            <Button variant="ghost" size="icon" onClick={onBack} className="h-10 w-10 rounded-full">
              <ChevronLeft size={20} />
            </Button>
          )}

          {onMenu && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onMenu}
              className="h-10 w-10 rounded-full lg:hidden"
            >
              <Menu size={20} />
            </Button>
          )}

          {/* Title section */}
          <div className="flex-1">
            <h1 className="text-lg font-semibold truncate">{title}</h1>
            {subtitle && <p className="text-xs text-muted-foreground truncate">{subtitle}</p>}
          </div>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-2">
          {onSettings && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onSettings}
              className="h-10 w-10 rounded-full"
            >
              <Settings size={18} />
            </Button>
          )}
        </div>
      </div>

      {/* Progress indicator for loading states */}
      <motion.div
        className="absolute bottom-0 left-0 h-0.5 bg-primary"
        initial={{ width: 0 }}
        animate={{ width: isVisible ? '0%' : '100%' }}
        transition={{ duration: 0.3 }}
      />
    </motion.header>
  );
};

// Companion hook for header visibility
export const useMobileChatHeader = () => {
  const [isVisible, setIsVisible] = React.useState(true);
  const scrollThreshold = 50;
  const lastScrollYRef = React.useRef(0);

  React.useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      const lastScrollY = lastScrollYRef.current;

      // Show header when scrolling up or near the top
      if (currentScrollY < lastScrollY || currentScrollY < scrollThreshold) {
        setIsVisible(true);
      }
      // Hide header when scrolling down past the threshold
      else if (currentScrollY > lastScrollY && currentScrollY > scrollThreshold) {
        setIsVisible(false);
      }

      lastScrollYRef.current = currentScrollY;
    };

    let ticking = false;
    const throttledScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', throttledScroll, { passive: true });
    return () => window.removeEventListener('scroll', throttledScroll);
  }, [scrollThreshold]);

  return { isVisible, setIsVisible };
};

export default MobileChatHeader;
