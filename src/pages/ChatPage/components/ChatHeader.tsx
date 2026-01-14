import React, { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Brain, Zap, Layers, Sun, Moon } from 'lucide-react';
import { EnhancedBackButton } from '@/components/ui/enhanced-back-button';
import { TripPlanner } from '@/components/TripPlanner';
import { HelpDialog } from '@/pages/ChatPage/components/HelpDialog';
import { WhatsNewModal } from '@/pages/ChatPage/components/WhatsNewModal';
import HowItWorksModal from '@/pages/ChatPage/components/HowItWorksModal';
import { WHATS_NEW_VERSION } from '@/pages/ChatPage/constants/whatsNew';
import { HamburgerMenu } from '@/components/HamburgerMenu';
import LogoImage from '@/components/LogoImage';
import { Button } from '@/components/ui/button';
import { StorageKeys } from '@/constants/storage';
import { useLocalStorage } from '@/hooks/useLocalStorage';

interface ChatHeaderProps {
  theme: string;
  toggleTheme: () => void;
  modelMode: 'fast' | 'smart';
  setModelMode: (mode: 'fast' | 'smart') => void;
  onTripPlanSubmit: (plan: string) => void;
  shortAnswerMode: boolean;
  setShortAnswerMode: (value: boolean) => void;
  onExportMarkdown: () => void;
  onClearConversation: () => void;
  onInsertExample?: (text: string) => void;
  menuOpen: boolean;
  setMenuOpen: (value: boolean) => void;
  highlightModelMode: boolean;
  highlightShortAnswers: boolean;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  theme,
  toggleTheme,
  modelMode,
  setModelMode,
  onTripPlanSubmit,
  shortAnswerMode,
  setShortAnswerMode,
  onExportMarkdown,
  onClearConversation,
  onInsertExample,
  menuOpen,
  setMenuOpen,
  highlightModelMode,
  highlightShortAnswers,
}) => {
  const [showTripPlanner, setShowTripPlanner] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [showWhatsNew, setShowWhatsNew] = useState(false);
  const [showHowItWorks, setShowHowItWorks] = useState(false);

  // Track if there's unseen updates
  // We keep this local to the header since it owns the modal in this component
  const [lastSeenVersion, setLastSeenVersion] = useLocalStorage<string>(
    StorageKeys.whatsNewLastSeen,
    '',
  );
  const hasWhatsNew = lastSeenVersion !== WHATS_NEW_VERSION;
  const prefersReducedMotion = useReducedMotion();

  useEffect(() => {
    if (hasWhatsNew) {
      // Open once on load if there are unseen updates
      setShowWhatsNew(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <motion.header
        className="border-b border-[var(--border)] glass backdrop-blur-xl sticky top-0 z-40 shadow-sm max-w-full overflow-hidden"
        initial={prefersReducedMotion ? undefined : { y: -100 }}
        animate={prefersReducedMotion ? undefined : { y: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      >
        <div className="h-14 px-2 sm:px-4 flex items-center justify-between max-w-full">
          <div className="flex items-center">
            <EnhancedBackButton to="/" label="Home" variant="minimal" size="sm" />
            <div className="h-6 w-px bg-border/50 mx-2 sm:mx-3" />
            <motion.div className="mr-2 sm:mr-3 h-6 sm:h-7 md:h-8" whileHover={{ scale: 1.05 }}>
              <LogoImage fitParent className="h-full w-auto" />
            </motion.div>
            <span className="text-base sm:text-xl md:text-2xl font-bold text-foreground">
              32 CBG <span className="hidden sm:inline">Policy Assistant</span>
            </span>
          </div>
          <motion.div
            className="flex items-center gap-2"
            initial={prefersReducedMotion ? undefined : { opacity: 0, x: 20 }}
            animate={prefersReducedMotion ? undefined : { opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Button
              variant="outline"
              size="icon"
              onClick={toggleTheme}
              className="h-11 w-11 rounded-lg border-2 shadow-md hover:shadow-lg transition-all duration-200 bg-[var(--card)] hover:bg-[var(--accent)] hover:border-[var(--accent-foreground)]"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>
            <HamburgerMenu
              theme={theme}
              toggleTheme={toggleTheme}
              modelMode={modelMode}
              setModelMode={setModelMode}
              shortAnswerMode={shortAnswerMode}
              setShortAnswerMode={setShortAnswerMode}
              onTripPlannerOpen={() => setShowTripPlanner(true)}
              onWhatsNewOpen={() => setShowWhatsNew(true)}
              onHowItWorksOpen={() => setShowHowItWorks(true)}
              onHelpOpen={() => setShowHelp(true)}
              onExportMarkdown={onExportMarkdown}
              onClearConversation={onClearConversation}
              hasWhatsNew={hasWhatsNew}
              isOpen={menuOpen}
              onOpenChange={setMenuOpen}
              highlightModelMode={highlightModelMode}
              highlightShortAnswers={highlightShortAnswers}
            />
          </motion.div>
        </div>
      </motion.header>

      {/* Modals */}
      <TripPlanner
        open={showTripPlanner}
        onOpenChange={setShowTripPlanner}
        onSubmit={(plan) => {
          onTripPlanSubmit(plan);
          setShowTripPlanner(false);
        }}
      />

      {/* What's New */}
      <WhatsNewModal
        open={showWhatsNew}
        onOpenChange={(open) => {
          if (!open) {
            setLastSeenVersion(WHATS_NEW_VERSION);
          }
          setShowWhatsNew(open);
        }}
      />

      <HowItWorksModal open={showHowItWorks} onOpenChange={setShowHowItWorks} />

      <HelpDialog open={showHelp} onOpenChange={setShowHelp} onInsertExample={onInsertExample} />
    </>
  );
};
