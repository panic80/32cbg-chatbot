import { FormEvent, ReactNode, UIEvent, useCallback, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ChevronDown, Send } from 'lucide-react';
import LogoImage from '@/components/LogoImage';

import '@/styles/landing.css';
import '@/styles/sticky-footer.css';
import { SITE_CONFIG, getCopyrightText, getLastUpdatedText } from '@/constants/siteConfig';
import { useTheme } from '@/context/ThemeContext';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { FeatureCard } from '@/components/ui/feature-card';
import { useCopyToClipboard } from '@/hooks/use-copy-to-clipboard';
import { cn } from '@/lib/utils';
import { footerLinks, landingFeatures, quickAskPrompts } from './landingConfig';

type ModalType = 'privacy' | 'about' | 'scip' | null;

interface ScrollableModalProps {
  children: ReactNode;
  showScrollIndicator: boolean;
  onScroll: (event: UIEvent<HTMLDivElement>) => void;
  contentClassName?: string;
}

function ScrollableModal({
  children,
  showScrollIndicator,
  onScroll,
  contentClassName,
}: ScrollableModalProps): JSX.Element {
  return (
    <div className="relative">
      <div className={cn('overflow-y-auto max-h-[60vh] pr-2', contentClassName)} onScroll={onScroll}>
        {children}
      </div>
      {showScrollIndicator && (
        <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-[var(--background)] to-transparent pointer-events-none flex items-end justify-center pb-2">
          <ChevronDown className="w-5 h-5 text-[var(--primary)] animate-bounce" />
        </div>
      )}
    </div>
  );
}

const CLOSE_BUTTON_CLASS =
  'w-full px-4 py-2 sm:py-3 text-center text-sm sm:text-base text-[var(--text)] bg-[var(--card)] hover:bg-[var(--primary)] hover:text-white rounded-lg transition-colors duration-300';

function LandingPage(): JSX.Element {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const [query, setQuery] = useState('');
  const [activeModal, setActiveModal] = useState<ModalType>(null);
  const [isNavigatingToSCIP, setIsNavigatingToSCIP] = useState(false);
  const [privacyScrollIndicator, setPrivacyScrollIndicator] = useState(true);
  const [aboutScrollIndicator, setAboutScrollIndicator] = useState(true);

  const { isCopied: isLinkCopied, handleCopy: copySCIPLink } = useCopyToClipboard({
    text: SITE_CONFIG.SCIP_PORTAL_URL,
    copyMessage: 'SCIP link copied to clipboard',
  });

  const closeModal = useCallback(() => setActiveModal(null), []);

  const handleScrollCheck = useCallback(
    (event: UIEvent<HTMLDivElement>, setIndicator: (value: boolean) => void) => {
      const element = event.currentTarget;
      const isAtBottom = element.scrollHeight - element.scrollTop <= element.clientHeight + 10;
      setIndicator(!isAtBottom);
    },
    [],
  );

  const handleAskSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = query.trim();
      navigate(trimmed.length === 0 ? '/chat' : `/chat?q=${encodeURIComponent(trimmed)}`);
    },
    [navigate, query],
  );

  const quickAsk = useCallback(
    (prompt: string) => {
      navigate(`/chat?q=${encodeURIComponent(prompt)}`);
    },
    [navigate],
  );

  const confirmSCIPNavigation = useCallback(() => {
    if (isNavigatingToSCIP) return;
    setIsNavigatingToSCIP(true);
    setActiveModal(null);
    window.location.assign(SITE_CONFIG.SCIP_PORTAL_URL);
  }, [isNavigatingToSCIP]);

  const handleFooterLink = useCallback((id: (typeof footerLinks)[number]['id']) => {
    if (id === 'about') {
      setActiveModal('about');
    } else if (id === 'privacy') {
      setActiveModal('privacy');
    }
  }, []);

  const renderFeatureCard = useCallback(
    (feature: (typeof landingFeatures)[number]) => {
      const commonProps = {
        title: feature.title,
        description: feature.description,
        icon: feature.icon,
      };

      // Resources card with "Coming Soon" overlay
      if (feature.id === 'resources' && feature.to) {
        return (
          <a
            key={feature.id}
            href={feature.to}
            className={cn('lpt-minimal-card', 'lpt-minimal-card-disabled', 'relative')}
            title={feature.description}
            aria-label={`${feature.title} - ${feature.description}`}
            style={{ pointerEvents: 'auto' }}
          >
            <FeatureCard variant="minimal" {...commonProps} badge={feature.badge} />
            <div className="absolute inset-0 bg-[var(--background)]/50 rounded-2xl z-10 flex items-center justify-center pointer-events-none">
              <div className="bg-[var(--primary)] text-white px-4 py-2 rounded-full font-medium text-sm shadow-lg animate-pulse">
                {feature.badge}
              </div>
            </div>
          </a>
        );
      }

      // Link cards
      if (feature.kind === 'link' && feature.to) {
        return (
          <Link
            key={feature.id}
            to={feature.to}
            className="lpt-minimal-card"
            title={feature.description}
            aria-label={`${feature.title} - ${feature.description}`}
          >
            <FeatureCard variant="minimal" {...commonProps} />
          </Link>
        );
      }

      // Action cards (SCIP)
      if (feature.kind === 'action') {
        return (
          <button
            key={feature.id}
            type="button"
            onClick={() => setActiveModal('scip')}
            className="lpt-minimal-card"
            title={feature.description}
            aria-label={`${feature.title} - ${feature.description}`}
          >
            <FeatureCard variant="minimal" {...commonProps} />
          </button>
        );
      }

      // Disabled cards
      if (feature.kind === 'disabled') {
        const disabledMessage =
          feature.disabledTooltip ??
          `${feature.title} is currently unavailable. We're working to restore access soon.`;

        return (
          <div
            key={feature.id}
            className={cn('lpt-minimal-card', 'lpt-minimal-card-disabled')}
            aria-disabled="true"
            title={disabledMessage}
            aria-label={`${feature.title} - ${feature.description}. ${disabledMessage}`}
            data-disabled-tooltip={disabledMessage}
          >
            <FeatureCard
              variant="minimal"
              {...commonProps}
              badge={feature.badge}
              disabled
              disabledLabel={feature.badge}
            />
          </div>
        );
      }

      // Default cards
      return (
        <div key={feature.id} className="lpt-minimal-card">
          <FeatureCard variant="minimal" {...commonProps} />
        </div>
      );
    },
    [],
  );

  return (
    <div className="lpt-minimal-root">
      <div className="lpt-minimal-bg" aria-hidden="true" />

      <button
        onClick={toggleTheme}
        className="lpt-minimal-theme"
        aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
      >
        {theme === 'light' ? (
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M20 14.12A7.78 7.78 0 019.88 4a7.78 7.78 0 002.9 15.1 7.78 7.78 0 007.22-5z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
            <path
              d="M12 2v2m0 16v2M2 12h2m16 0h2m-3-7l-1.5 1.5M4.93 4.93l1.5 1.5m11.14 11.14l1.5 1.5M4.93 19.07l1.5-1.5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        )}
      </button>

      <div className="lpt-minimal-content">
        <div className="lpt-minimal-hero">
          <div className="lpt-minimal-logo">
            <LogoImage size="xl" />
          </div>

          <h1 className="lpt-minimal-title">32 CBG G8 Administration Hub</h1>
          <p className="lpt-minimal-subtitle">Comprehensive Gateway to Financial Resources</p>

          <form onSubmit={handleAskSubmit} className="lpt-minimal-search">
            <div className="lpt-minimal-search-wrapper">
              <div className="lpt-minimal-search-box">
                <input
                  type="text"
                  className="lpt-minimal-search-input"
                  placeholder="Ask a policy question..."
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  aria-label="Ask a policy question"
                />
                <button type="submit" className="lpt-minimal-search-btn">
                  <span>Ask</span>
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </form>

          <div className="lpt-minimal-chips">
            {quickAskPrompts.map((prompt) => (
              <button
                key={prompt.query}
                type="button"
                className="lpt-minimal-chip"
                onClick={() => quickAsk(prompt.query)}
              >
                {prompt.label}
              </button>
            ))}
          </div>

          <div className="lpt-minimal-cards">{landingFeatures.map(renderFeatureCard)}</div>
        </div>
      </div>

      <footer className="lpt-minimal-footer" role="contentinfo">
        <div className="lpt-minimal-footer-links">
          {footerLinks.map((link) =>
            link.id === 'contact' ? (
              <a
                key={link.id}
                href={`mailto:${SITE_CONFIG.CONTACT_EMAIL}?subject=Contacting%20from%20G8%20homepage`}
                className="lpt-minimal-footer-link"
              >
                <link.icon className="w-3.5 h-3.5" aria-hidden="true" />
                <span>{link.label}</span>
              </a>
            ) : (
              <button
                key={link.id}
                type="button"
                onClick={() => handleFooterLink(link.id)}
                className="lpt-minimal-footer-link"
              >
                <link.icon className="w-3.5 h-3.5" aria-hidden="true" />
                <span>{link.label}</span>
              </button>
            ),
          )}
        </div>

        <div className="lpt-minimal-footer-meta">
          <p>{getCopyrightText()}</p>
          <p style={{ marginTop: '0.25rem' }}>{getLastUpdatedText()}</p>
        </div>
      </footer>

      {/* Privacy Modal */}
      <Dialog open={activeModal === 'privacy'} onOpenChange={(open) => !open && closeModal()}>
        <DialogContent className="max-w-[32rem] max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>Privacy Policy</DialogTitle>
          </DialogHeader>
          <ScrollableModal
            showScrollIndicator={privacyScrollIndicator}
            onScroll={(e) => handleScrollCheck(e, setPrivacyScrollIndicator)}
            contentClassName="space-y-4 sm:space-y-6"
          >
            <h3 className="text-base sm:text-lg font-semibold">General Privacy Notice</h3>
            <p className="text-sm sm:text-base text-[var(--text)] leading-relaxed">
              We prioritize the protection of your personal information and are committed to
              maintaining your trust.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mt-4 sm:mt-6">
              Data Collection &amp; Usage
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-[var(--text)] opacity-80">
              <li>We collect only essential information needed for the service</li>
              <li>Your data is encrypted and stored securely</li>
              <li>We do not sell or share your personal information</li>
              <li>You have control over your data and can request its deletion</li>
            </ul>
            <h3 className="text-base sm:text-lg font-semibold mt-4 sm:mt-6">
              AI Processing (OpenAI)
            </h3>
            <p className="text-sm sm:text-base text-[var(--text)] leading-relaxed">
              This application uses OpenAI&apos;s GPT models. When you interact with our AI
              features:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-[var(--text)] opacity-80">
              <li>Your conversations may be processed to improve responses</li>
              <li>No personally identifiable information is retained by the AI</li>
              <li>Conversations are not used to train the core AI model</li>
              <li>You can opt out of AI features at any time</li>
            </ul>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-4 sm:mt-6">
              For more details about OpenAI&apos;s data handling, please visit OpenAI&apos;s privacy
              policy.
            </p>
            <div className="pt-2">
              <button onClick={closeModal} className={CLOSE_BUTTON_CLASS}>
                Close
              </button>
            </div>
          </ScrollableModal>
        </DialogContent>
      </Dialog>

      {/* About Modal */}
      <Dialog open={activeModal === 'about'} onOpenChange={(open) => !open && closeModal()}>
        <DialogContent className="max-w-lg max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>About This Page</DialogTitle>
          </DialogHeader>
          <ScrollableModal
            showScrollIndicator={aboutScrollIndicator}
            onScroll={(e) => handleScrollCheck(e, setAboutScrollIndicator)}
          >
            <h3 className="text-base sm:text-lg font-semibold mb-2 text-[var(--primary)]">
              32 CBG G8 Admin Hub
            </h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base">
              A comprehensive digital platform designed to streamline administrative processes for
              Canadian Armed Forces personnel, with a focus on travel claims, policy guidance, and
              financial services.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Key Features</h3>
            <ul className="list-disc list-inside mb-3 sm:mb-4 text-sm sm:text-base space-y-1">
              <li>
                <strong className="text-[var(--primary)]">Policy Assistant</strong> - AI-powered
                chatbot providing instant guidance
              </li>
              <li>
                <strong className="text-[var(--primary)]">SCIP Portal</strong> - Direct access to
                claims submission platform
              </li>
              <li>
                <strong className="text-[var(--primary)]">OPI Contacts</strong> - Comprehensive
                directory of FSC and FMC personnel
              </li>
              <li>
                <strong className="text-[var(--primary)]">Resources</strong> - Consolidated SOPs,
                guides, and templates for day-to-day administration
              </li>
            </ul>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Disclaimer</h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base text-[var(--text-secondary)]">
              This is an unofficial site not affiliated with DND, CAF, or any government department.
              Information provided is for reference only. Always verify critical information through
              official channels.
            </p>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-4 pt-4 border-t border-[var(--border)]">
              Maintained by the 32 CBG G8 Team
            </p>
            <div className="pt-4">
              <button onClick={closeModal} className={CLOSE_BUTTON_CLASS}>
                Close
              </button>
            </div>
          </ScrollableModal>
        </DialogContent>
      </Dialog>

      {/* SCIP Confirmation Modal */}
      <Dialog open={activeModal === 'scip'} onOpenChange={(open) => !open && closeModal()}>
        <DialogContent className="w-[92vw] sm:max-w-lg md:max-w-xl lg:max-w-2xl break-words">
          <DialogHeader>
            <DialogTitle>SCIP Portal</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm sm:text-base break-words">
              You are about to navigate to the SCIP Portal, which is an external Microsoft PowerApps
              platform. Have your D365 login (@ecn.forces.gc.ca) ready.
            </p>
            <p className="text-sm sm:text-base text-[var(--text-secondary)] break-words">
              This will open in a new tab. Do you want to continue?
            </p>
            <div className="mb-2 p-3 bg-[var(--background-secondary)] rounded-lg border border-[var(--border)] w-full">
              <p className="text-xs sm:text-sm text-[var(--text-secondary)] mb-3">
                If the portal does not open, please copy the URL below and paste it directly into
                your browser:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-[minmax(0,1fr)_auto] gap-2 items-center">
                <div className="min-w-0 w-full p-2 bg-[var(--background)] rounded text-xs font-mono text-[var(--text-secondary)] overflow-hidden">
                  <div className="block truncate max-w-full">
                    {SITE_CONFIG.SCIP_PORTAL_URL.substring(0, 50)}...
                  </div>
                </div>
                <button
                  onClick={copySCIPLink}
                  disabled={isLinkCopied}
                  className={cn(
                    'px-3 py-2 text-xs sm:text-sm rounded-lg transition-all duration-300 flex items-center gap-2 whitespace-nowrap shrink-0 mt-2 sm:mt-0 justify-center',
                    isLinkCopied
                      ? 'bg-green-600/20 text-green-600 cursor-not-allowed'
                      : 'bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)]',
                  )}
                >
                  {isLinkCopied ? 'Link Copied' : 'Copy Link'}
                </button>
              </div>
            </div>
            <div className="flex gap-3 justify-end">
              <button
                onClick={closeModal}
                className="px-4 py-2 text-sm sm:text-base text-[var(--text)] bg-[var(--background-secondary)] hover:bg-[var(--background)] rounded-lg transition-colors duration-300"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmSCIPNavigation}
                disabled={isNavigatingToSCIP}
                className={cn(
                  'px-4 py-2 text-sm sm:text-base rounded-lg transition-colors duration-300',
                  isNavigatingToSCIP
                    ? 'bg-[var(--primary)]/60 text-white cursor-not-allowed'
                    : 'text-white bg-[var(--primary)] hover:bg-[var(--primary-hover)]',
                )}
              >
                {isNavigatingToSCIP ? 'Opening...' : 'Continue'}
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default LandingPage;
