import React, { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Search,
  Users,
  Mail as MailIcon,
  Building2 as Building,
  Info,
  ShieldCheck,
  Sun,
  Moon,
} from 'lucide-react';
import '../styles/landing.css';
import '../styles/sticky-footer.css';
import { SITE_CONFIG, getCopyrightText, getLastUpdatedText } from '../constants/siteConfig';
import LogoImage from '@/components/LogoImage';

// shadcn/ui components
import { Button } from '@/components/ui/button';
import { AnimatedButton } from '@/components/ui/animated-button';
import { EnhancedBackButton } from '@/components/ui/enhanced-back-button';
import { cn } from '@/lib/utils';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { useTheme } from '@/context/ThemeContext';

// Import ReimaginedOPIView
import ReimaginedOPIView from './OPIPage/ReimaginedOPIView';
import { forceScrollToTop, forceScrollToTopDeferred } from '@/utils/scroll';

export default function OPIPage() {
  const location = useLocation();
  const topRef = useRef(null);
  const { theme, toggleTheme } = useTheme();
  const [contactView, setContactView] = useState('all');
  const [selectedUnit, setSelectedUnit] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [showAboutModal, setShowAboutModal] = useState(false);

  // Ensure immediate and deferred scroll reset on page load
  useLayoutEffect(() => {
    try {
      if (topRef.current && 'scrollIntoView' in topRef.current) {
        topRef.current.scrollIntoView({ behavior: 'auto', block: 'start', inline: 'nearest' });
      }
    } catch {
      // scrollIntoView may fail in some browsers - forceScrollToTop handles fallback
    }
    forceScrollToTop();
    const cleanup = forceScrollToTopDeferred();
    return cleanup;
  }, []);

  // Simulate loading state
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  // Theme is managed globally by ThemeProvider

  // Handle About link click
  const handleAboutClick = (e) => {
    e.preventDefault();
    setShowAboutModal(true);
  };

  // Contact data
  const unitContacts = {
    '2 Int': {
      fsc: 'Cpl Downes',
      fscEmail: 'william.downes@forces.gc.ca',
      fmc: 'MCpl Angela McDonald',
      fmcEmail: 'ANGELA.MCDONALD@forces.gc.ca',
    },
    '32 CBG HQ': {
      fsc: 'Cpl Downes',
      fscEmail: 'william.downes@forces.gc.ca',
      fmc: 'Sgt Gordon Brown',
      fmcEmail: 'GORDON.BROWN2@forces.gc.ca',
    },
    '32 CER': {
      fsc: 'Cpl Downes',
      fscEmail: 'william.downes@forces.gc.ca',
      fmc: 'Sgt Gordon Brown',
      fmcEmail: 'GORDON.BROWN2@forces.gc.ca',
    },
    '32 Svc Bn': {
      fsc: 'Cpl Downes',
      fscEmail: 'william.downes@forces.gc.ca',
      fmc: 'Sgt Gordon Brown',
      fmcEmail: 'GORDON.BROWN2@forces.gc.ca',
    },
    GGHG: {
      fsc: 'Cpl Downes',
      fscEmail: 'william.downes@forces.gc.ca',
      fmc: 'Sgt Jennifer Wood',
      fmcEmail: 'JENNIFER.WOOD@forces.gc.ca',
    },
    '48th Highrs': {
      fsc: 'Sgt Ro',
      fscEmail: 'eugene.ro@forces.gc.ca',
      fmc: 'Sgt Jennifer Wood',
      fmcEmail: 'JENNIFER.WOOD@forces.gc.ca',
    },
    '7 Tor': {
      fsc: 'Sgt Ro',
      fscEmail: 'eugene.ro@forces.gc.ca',
      fmc: 'Sgt Jennifer Wood',
      fmcEmail: 'JENNIFER.WOOD@forces.gc.ca',
    },
    'Tor Scots': {
      fsc: 'Sgt Ro',
      fscEmail: 'eugene.ro@forces.gc.ca',
      fmc: 'MCpl Angela McDonald',
      fmcEmail: 'ANGELA.MCDONALD@forces.gc.ca',
    },
    QOR: {
      fsc: 'Sgt Ro',
      fscEmail: 'eugene.ro@forces.gc.ca',
      fmc: 'Sgt Gordon Brown',
      fmcEmail: 'GORDON.BROWN2@forces.gc.ca',
    },
    '32 Sig Regt': {
      fsc: 'Sgt Zeng',
      fscEmail: 'aidi.zeng@forces.gc.ca',
      fmc: 'MCpl Angela McDonald',
      fmcEmail: 'ANGELA.MCDONALD@forces.gc.ca',
    },
    'Lorne Scots': {
      fsc: 'Sgt Zeng',
      fscEmail: 'aidi.zeng@forces.gc.ca',
      fmc: 'MCpl Angela McDonald',
      fmcEmail: 'ANGELA.MCDONALD@forces.gc.ca',
    },
    'QY Rang': {
      fsc: 'Sgt Zeng',
      fscEmail: 'aidi.zeng@forces.gc.ca',
      fmc: 'Sgt Jennifer Wood',
      fmcEmail: 'JENNIFER.WOOD@forces.gc.ca',
    },
    'R Regt C': {
      fsc: 'Sgt Zeng',
      fscEmail: 'aidi.zeng@forces.gc.ca',
      fmc: 'Sgt Gordon Brown',
      fmcEmail: 'GORDON.BROWN2@forces.gc.ca',
    },
    'Linc & Welld': {
      fsc: 'N/A',
      fscEmail: '',
      fmc: 'Sgt Mabel James',
      fmcEmail: 'MABEL.JAMES@forces.gc.ca',
    },
    '56 Fd': {
      fsc: 'N/A',
      fscEmail: '',
      fmc: 'Sgt Mabel James',
      fmcEmail: 'MABEL.JAMES@forces.gc.ca',
    },
  };

  const allUnits = Object.keys(unitContacts).sort();
  const filteredUnits = allUnits.filter((unit) =>
    unit.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  // FSC contacts organized by section
  const fscContacts = [
    {
      name: 'PO 1 Salehi',
      role: 'FSC Warrant Officer',
      email: 'Amir.Salehi@forces.gc.ca',
      isLeadership: true,
    },
    {
      name: 'Sgt Zeng',
      role: 'FSC Second-in-Command',
      email: 'aidi.zeng@forces.gc.ca',
      isLeadership: true,
    },
    {
      name: 'Cpl Downes',
      role: 'FSC 1 Section',
      email: 'william.downes@forces.gc.ca',
      units: ['2 Int', '32 CBG HQ', '32 CER', '32 Svc Bn', 'GGHG'],
    },
    {
      name: 'Sgt Ro',
      role: 'FSC 2 Section',
      email: 'eugene.ro@forces.gc.ca',
      units: ['48th Highrs', '7 Tor', 'Tor Scots', 'QOR'],
    },
    {
      name: 'Sgt Zeng',
      role: 'FSC 3 Section',
      email: 'aidi.zeng@forces.gc.ca',
      units: ['32 Sig Regt', 'Lorne Scots', 'QY Rang', 'R Regt C'],
    },
  ];

  // FMC contacts organized by group
  const fmcContacts = [
    {
      name: 'Sgt Peter Cuprys',
      role: 'FMC Warrant Officer',
      email: 'PETER.CUPRYS@forces.gc.ca',
      units: [],
      isLeadership: true,
    },
    {
      name: 'Sgt Jennifer Wood',
      role: 'FMC 1 Section',
      email: 'JENNIFER.WOOD@forces.gc.ca',
      units: ['GGHG', 'QY Rang', '7 Tor', '48th Highrs'],
    },
    {
      name: 'Sgt Gordon Brown',
      role: 'FMC 2 Section',
      email: 'GORDON.BROWN2@forces.gc.ca',
      units: ['R Regt C', '32 Svc Bn', 'QOR', '32 CER'],
    },
    {
      name: 'MCpl Angela McDonald',
      role: 'FMC 3 Section',
      email: 'ANGELA.MCDONALD@forces.gc.ca',
      units: ['32 Sig Regt', 'Lorne Scots', 'Tor Scots', '2 Int'],
    },
    {
      name: 'Sgt Mabel James',
      role: 'FMC 4 Section',
      email: 'MABEL.JAMES@forces.gc.ca',
      units: ['Linc & Welld', '56 Fd'],
    },
  ];

  return (
    <div ref={topRef} className="root-container" style={{ scrollBehavior: 'auto' }}>
      <div className="flex flex-col min-h-screen">
        <div className="flex-grow">
          <div className="bg-[var(--background)] text-[var(--text)]">
            {/* Header */}
            <header className="border-b border-[var(--border)] glass backdrop-blur-xl sticky top-0 z-40 shadow-sm">
              <div className="h-16 px-3 sm:px-6 flex items-center justify-between">
                <div className="flex items-center gap-3 sm:gap-4">
                  <EnhancedBackButton to="/" label="Back" variant="minimal" size="sm" />
                  <div className="h-8 w-px bg-border/50" />
                  <div className="h-8 sm:h-9 md:h-10">
                    <LogoImage fitParent className="h-full w-auto" />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-base sm:text-xl md:text-2xl font-bold text-foreground">
                      <span className="sm:hidden">OPI</span>
                      <span className="hidden sm:inline">Office of Primary Interest</span>
                    </span>
                    <span className="text-xs text-muted-foreground hidden sm:block">
                      Contact Directory
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={toggleTheme}
                    className="h-11 w-11 rounded-lg border-2 shadow-md hover:shadow-lg transition-all duration-200 bg-[var(--card)] hover:bg-[var(--accent)] hover:border-[var(--accent-foreground)]"
                    aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  >
                    {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                  </Button>
                </div>
              </div>
            </header>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 xl:px-12 py-6 sm:py-12">
              {/* Reimagined View */}
              <ReimaginedOPIView
                unitContacts={unitContacts}
                fscContacts={fscContacts}
                fmcContacts={fmcContacts}
                contactView={contactView}
                selectedUnit={selectedUnit}
                searchTerm={searchTerm}
                setSelectedUnit={setSelectedUnit}
                setSearchTerm={setSearchTerm}
                setContactView={setContactView}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer
          className="mt-auto px-4 sm:px-6 lg:px-8 border-t border-[var(--border)]"
          role="contentinfo"
        >
          <div className="max-w-5xl mx-auto py-6">
            {/* Mobile-optimized footer content */}
            <div className="md:hidden">
              <nav className="flex justify-around my-2" aria-label="Footer Navigation">
                <button
                  type="button"
                  onClick={handleAboutClick}
                  className="inline-flex flex-col items-center text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1"
                >
                  <Info className="w-5 h-5" aria-hidden="true" />
                  <span className="text-xs mt-1">About</span>
                </button>
                <a
                  href="mailto:g8@sent.com?subject=Contacting%20from%20G8%20homepage"
                  className="inline-flex flex-col items-center text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1"
                >
                  <MailIcon className="w-5 h-5" aria-hidden="true" />
                  <span className="text-xs mt-1">Contact</span>
                </a>
                <button
                  type="button"
                  onClick={() => setShowPrivacyModal(true)}
                  className="inline-flex flex-col items-center text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1"
                >
                  <ShieldCheck className="w-5 h-5" aria-hidden="true" />
                  <span className="text-xs mt-1">Privacy</span>
                </button>
              </nav>
              <div className="text-center text-xs text-[var(--text)] opacity-50 mt-1">
                <p>{getCopyrightText()}</p>
              </div>
            </div>

            {/* Desktop footer content */}
            <div className="hidden md:block">
              <nav
                className="flex flex-wrap justify-center gap-4 sm:gap-6 lg:gap-8 mb-4"
                aria-label="Footer Navigation"
              >
                <button
                  type="button"
                  onClick={handleAboutClick}
                  className="inline-flex items-center space-x-2 text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1 text-sm sm:text-base"
                >
                  <Info className="w-4 h-4 sm:w-5 sm:h-5" aria-hidden="true" />
                  <span>About</span>
                </button>
                <a
                  href="mailto:g8@sent.com?subject=Contacting%20from%20G8%20homepage"
                  className="inline-flex items-center space-x-2 text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1 text-sm sm:text-base"
                >
                  <MailIcon className="w-4 h-4 sm:w-5 sm:h-5" aria-hidden="true" />
                  <span>Contact</span>
                </a>
                <button
                  type="button"
                  onClick={() => setShowPrivacyModal(true)}
                  className="inline-flex items-center space-x-2 text-[var(--text)] opacity-70 hover:opacity-100 hover:text-[var(--primary)] transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] rounded px-2 py-1 text-sm sm:text-base"
                >
                  <ShieldCheck className="w-4 h-4 sm:w-5 sm:h-5" aria-hidden="true" />
                  <span>Privacy Policy</span>
                </button>
              </nav>
              <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2 text-xs sm:text-sm text-[var(--text)] opacity-50">
                <p>{getCopyrightText()}</p>
                <p>{getLastUpdatedText()}</p>
              </div>
            </div>
          </div>
        </footer>
      </div>

      <Dialog open={showPrivacyModal} onOpenChange={setShowPrivacyModal}>
        <DialogContent className="max-w-[32rem] break-words">
          <DialogHeader>
            <DialogTitle>Privacy Policy</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 sm:space-y-6 break-words">
            <h3 className="text-base sm:text-lg font-semibold">General Privacy Notice</h3>
            <p className="text-sm sm:text-base text-[var(--text)] leading-relaxed break-words">
              We prioritize the protection of your personal information and are committed to
              maintaining your trust.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mt-4 sm:mt-6">
              Data Collection & Usage
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-[var(--text)] opacity-80">
              <li>We collect only essential information needed for the service</li>
              <li>Your data is encrypted and stored securely</li>
              <li>We do not sell or share your personal information</li>
              <li>You have control over your data and can request its deletion</li>
            </ul>
            <h3 className="text-base sm:text-lg font-semibold mt-4 sm:mt-6">
              AI Processing (Gemini)
            </h3>
            <p className="text-sm sm:text-base text-[var(--text)] leading-relaxed break-words">
              This application uses Google's Gemini AI. When you interact with our AI features:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-[var(--text)] opacity-80">
              <li>Your conversations may be processed to improve responses</li>
              <li>No personally identifiable information is retained by the AI</li>
              <li>Conversations are not used to train the core AI model</li>
              <li>You can opt out of AI features at any time</li>
            </ul>
            <div className="pt-2">
              <button
                onClick={() => setShowPrivacyModal(false)}
                className="w-full px-4 py-2 sm:py-3 text-center text-sm sm:text-base text-[var(--text)] bg-[var(--card)] hover:bg-[var(--primary)] hover:text-white rounded-lg transition-colors duration-200 h-10 sm:h-12"
              >
                Close
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={showAboutModal} onOpenChange={setShowAboutModal}>
        <DialogContent className="max-w-lg break-words">
          <DialogHeader>
            <DialogTitle>About This Page</DialogTitle>
          </DialogHeader>
          <div className="overflow-y-auto max-h-[calc(100vh-16rem)] break-words">
            <p className="mb-3 sm:mb-4 text-sm sm:text-base break-words">
              This unofficial site is not affiliated with the Department of National Defence (DND),
              the Canadian Armed Forces (CAF), or any associated departments or services. Use of
              this site is entirely at your own discretion.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Purpose</h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base break-words">
              Our goal is to provide Primary Reserve (P Res) members with quick and convenient
              access to essential G8 resources. We strive to streamline administrative processes and
              ensure you can locate accurate, up-to-date information whenever you need it.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Currently Available</h3>
            <ul className="list-disc list-inside mb-3 sm:mb-4 text-sm sm:text-base space-y-1">
              <li>
                <Link to="/chat" className="text-[var(--primary)] hover:underline">
                  Unofficial Policy Chatbot
                </Link>{' '}
                &ndash; An interactive tool designed to answer your questions about claims and
                travel entitlements, referencing the CFTDTI and NJC websites
              </li>
              <li>
                SCIP &ndash; Your centralized portal for financial and administrative functions
              </li>
              <li>SOPs &ndash; Standard Operating Procedures for day-to-day reference</li>
              <li>
                Onboarding Guide &ndash; A step-by-step manual to welcome and orient new members
              </li>
            </ul>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Privacy & Contact</h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base break-words">
              For privacy concerns, please use the Contact button or refer to our Privacy Policy.
              Your feedback is always welcome, and we look forward to improving your administrative
              experience.
            </p>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)]">
              Disclaimer: This page is not supported by the Defence Wide Area Network (DWAN).
            </p>
            <div className="pt-4">
              <button
                onClick={() => setShowAboutModal(false)}
                className="w-full px-4 py-2 sm:py-3 text-center text-sm sm:text-base text-[var(--text)] bg-[var(--card)] hover:bg-[var(--primary)] hover:text-white rounded-lg transition-colors duration-200 h-10 sm:h-12"
              >
                Close
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
