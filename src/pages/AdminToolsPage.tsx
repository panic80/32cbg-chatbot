import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  FileText,
  FolderOpen,
  GraduationCap,
  Wrench,
  BookOpen,
  Info,
  Mail,
  ShieldCheck,
} from 'lucide-react';
import '../styles/landing.css';
import '../styles/sticky-footer.css';
import { SITE_CONFIG, getCopyrightText, getLastUpdatedText } from '../constants/siteConfig';

// shadcn/ui components
import { Button } from '@/components/ui/button';
import { AnimatedButton } from '@/components/ui/animated-button';
import { EnhancedBackButton } from '@/components/ui/enhanced-back-button';
import { cn } from '@/lib/utils';

// Import FluentAdminView
import FluentAdminView from './AdminToolsPage/FluentAdminView';
// Import TabBasedAdminView
import TabBasedAdminView from './AdminToolsPage/TabBasedAdminView';
// Import DigitalLibraryView
import DigitalLibraryView from './AdminToolsPage/DigitalLibraryView';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';

export default function AdminToolsPage() {
  const location = useLocation();
  const topRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [showAboutModal, setShowAboutModal] = useState(false);
  // Scroll handling centralized by global ScrollToTop

  // Simulate loading state
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  // Handle About link click
  const handleAboutClick = (e) => {
    e.preventDefault();
    setShowAboutModal(true);
  };

  // Admin tools data
  const adminTools = [
    {
      id: 'sops',
      title: 'Standard Operating Procedures',
      description:
        'Access comprehensive SOPs and operational guidelines for daily administrative tasks',
      icon: FileText,
      links: [
        { name: 'SOP 001 - Leave Administration', url: '#' },
        { name: 'SOP 002 - Travel Claims Processing', url: '#' },
        { name: 'SOP 003 - Financial Reviews', url: '#' },
        { name: 'General Guidelines', url: '#' },
      ],
    },
    {
      id: 'onboarding',
      title: 'Onboarding Guide',
      description: 'Step-by-step resources to welcome and orient new unit members effectively',
      icon: GraduationCap,
      links: [
        { name: 'New Member Checklist', url: '#' },
        { name: 'D365 Account Setup', url: '#' },
        { name: 'SCIP Registration Guide', url: '#' },
        { name: 'First Claim Walkthrough', url: '#' },
      ],
    },
    {
      id: 'forms',
      title: 'Forms & Templates',
      description: 'Commonly used forms and document templates for administrative processes',
      icon: FolderOpen,
      links: [
        { name: 'CF 52 - Leave Request', url: '#' },
        { name: 'Travel Claim Template', url: '#' },
        { name: 'Memo Templates', url: '#' },
        { name: 'All Forms Directory', url: '#' },
      ],
    },
    {
      id: 'resources',
      title: 'Additional Resources',
      description: 'Quick links to external systems and reference materials',
      icon: Wrench,
      links: [
        {
          name: 'CFTDTI Official Site',
          url: 'https://www.canada.ca/en/department-national-defence/services/benefits-military/pay-pension-benefits/benefits/canadian-forces-temporary-duty-travel-instructions.html',
          external: true,
        },
        {
          name: 'NJC Travel Directive',
          url: 'https://www.njc-cnm.gc.ca/directive/d10/en',
          external: true,
        },
        { name: 'Monitor Mass Portal', url: '#' },
        { name: 'DWAN Resources', url: '#' },
      ],
    },
  ];

  return (
    <div ref={topRef} className="root-container" style={{ scrollBehavior: 'auto' }}>
      <div className="flex flex-col min-h-screen">
        <div className="flex-grow">
          <div className="bg-[var(--background)] text-[var(--text)]">
            {/* Header */}
            <header className="border-b border-[var(--border)] bg-background/95 backdrop-blur sticky top-0 z-40">
              <div className="h-14 px-4 flex items-center justify-between">
                <div className="flex items-center">
                  <EnhancedBackButton to="/" label="Back" variant="minimal" size="sm" />
                  <div className="h-6 w-px bg-border/50 mx-3" />
                  <BookOpen className="w-8 h-8 text-[var(--primary)] mr-2" />
                  <span className="text-2xl font-bold text-foreground">Resource Library</span>
                </div>
                <Button asChild variant="outline" size="sm" className="hidden sm:inline-flex">
                  <Link to="/admin/performance">Performance Dashboard</Link>
                </Button>
              </div>
            </header>

            <div className="w-full">
              {/* Digital Library View */}
              <DigitalLibraryView adminTools={adminTools} />
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
                  <EnvelopeIcon className="w-5 h-5" aria-hidden="true" />
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
              <div className="text-center text-xs text-[var(--text)] opacity-50">
                <p>{getCopyrightText()}</p>
                <p className="mt-1">{getLastUpdatedText()}</p>
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
                  <EnvelopeIcon className="w-4 h-4 sm:w-5 sm:h-5" aria-hidden="true" />
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

      {/* Privacy Modal */}
      {showPrivacyModal && (
        <>
          <div
            className="fixed inset-0 bg-black/60 z-40 animate-fade-in"
            onClick={() => setShowPrivacyModal(false)}
          />
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 animate-float-up">
            <div className="w-[min(90vw,_32rem)] bg-[var(--card)] text-[var(--text)] rounded-xl border border-[var(--border)] shadow-2xl overflow-hidden">
              <div className="p-4 sm:p-6 border-b border-[var(--border)]">
                <div className="flex justify-between items-center">
                  <h2 className="text-xl sm:text-2xl font-semibold">Privacy Policy</h2>
                  <button
                    onClick={() => setShowPrivacyModal(false)}
                    className="p-2 hover:bg-[var(--background-secondary)] rounded-full transition-colors w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center"
                    aria-label="Close privacy modal"
                  >
                    <svg
                      className="w-4 h-4 sm:w-5 sm:h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>
              </div>

              <div className="overflow-y-auto" style={{ maxHeight: 'calc(100vh - 16rem)' }}>
                <div className="p-4 sm:p-6 space-y-4 sm:space-y-6 break-words">
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
                    This application uses Google's Gemini AI. When you interact with our AI
                    features:
                  </p>
                  <ul className="list-disc pl-5 space-y-2 text-sm sm:text-base text-[var(--text)] opacity-80">
                    <li>Your conversations may be processed to improve responses</li>
                    <li>No personally identifiable information is retained by the AI</li>
                    <li>Conversations are not used to train the core AI model</li>
                    <li>You can opt out of AI features at any time</li>
                  </ul>

                  <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-4 sm:mt-6 break-words">
                    For more details about Gemini's data handling, please visit Google's AI privacy
                    policy.
                  </p>
                </div>
              </div>

              <div className="p-4 sm:p-6 border-t border-[var(--border)] bg-[var(--background-secondary)] rounded-b-xl">
                <button
                  onClick={() => setShowPrivacyModal(false)}
                  className="w-full px-4 py-2 sm:py-3 text-center text-sm sm:text-base text-[var(--text)] bg-[var(--card)] hover:bg-[var(--primary)] hover:text-white rounded-lg transition-colors duration-200 h-10 sm:h-12"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* About Modal */}
      <Dialog open={showAboutModal} onOpenChange={setShowAboutModal}>
        <DialogContent className="max-w-lg break-words">
          <DialogHeader>
            <DialogTitle>About This Page</DialogTitle>
          </DialogHeader>
          <div className="overflow-y-auto max-h-[calc(100vh-16rem)] break-words">
            <h3 className="text-base sm:text-lg font-semibold mb-2 text-[var(--primary)]">
              32 CBG G8 Resource Library
            </h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base break-words">
              A comprehensive digital platform designed to streamline administrative processes for
              Canadian Armed Forces personnel, with a focus on travel claims, policy guidance, and
              financial services.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Key Features</h3>
            <ul className="list-disc list-inside mb-3 sm:mb-4 text-sm sm:text-base space-y-1">
              <li>
                <strong className="text-[var(--primary)]">Policy Assistant</strong> – AI-powered
                chatbot providing instant guidance on CFTDTI policies, travel claims, and
                administrative procedures
              </li>
              <li>
                <strong className="text-[var(--primary)]">SCIP Portal</strong> – Direct access to
                the Streamlined Claims Interface Platform for digital claim submission
              </li>
              <li>
                <strong className="text-[var(--primary)]">OPI Contacts</strong> – Comprehensive
                directory of Financial Services (FSC) and Financial Management (FMC) personnel
                across 32 CBG units
              </li>
              <li>
                <strong className="text-[var(--primary)]">Resource Library</strong> – SOPs, how-to
                guides, FAQs, templates, and comprehensive administrative documentation
              </li>
            </ul>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Built for Efficiency</h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base break-words">
              This portal consolidates multiple resources into a single, user-friendly interface,
              reducing the time spent searching for information and ensuring consistent access to
              up-to-date policies and contacts.
            </p>
            <h3 className="text-base sm:text-lg font-semibold mb-2">Disclaimer</h3>
            <p className="mb-3 sm:mb-4 text-sm sm:text-base text-[var(--text-secondary)] break-words">
              This is an unofficial site not affiliated with DND, CAF, or any government department.
              Information provided is for reference only. Always verify critical information through
              official channels.
            </p>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)]">
              Not supported by the Defence Wide Area Network (DWAN). Use at your own discretion.
            </p>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-4 pt-4 border-t border-[var(--border)]">
              Maintained by the 32 CBG G8 Team
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
