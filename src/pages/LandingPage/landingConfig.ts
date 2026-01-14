import type { LucideIcon } from 'lucide-react';
import { CircleHelp, FileText, Users, Zap, Info, Mail, ShieldCheck } from 'lucide-react';

export type LandingFeatureKind = 'link' | 'action' | 'disabled';

export interface QuickAskPrompt {
  label: string;
  query: string;
}

export interface LandingFeature {
  id: string;
  title: string;
  description: string;
  icon: LucideIcon;
  kind: LandingFeatureKind;
  to?: string;
  badge?: string;
  disabledTooltip?: string;
}

export interface LandingFooterLink {
  id: 'about' | 'contact' | 'privacy';
  label: string;
  icon: LucideIcon;
}

export const quickAskPrompts: QuickAskPrompt[] = [
  {
    label: 'Mileage rates',
    query: 'What are the current mileage rates under CFTDTI?',
  },
  {
    label: 'Per diem rates',
    query: 'What are the meal per diem rates?',
  },
  {
    label: 'Travel advance',
    query: 'How do I request a travel advance?',
  },
  {
    label: 'Receipt requirements',
    query: 'What receipts do I need for claims?',
  },
];

export const landingFeatures: LandingFeature[] = [
  {
    id: 'policyAssistant',
    title: 'Policy Assistant',
    description:
      'Interactive, RAG powered AI chat to answer travel, benefits, and finance policy questions.',
    icon: CircleHelp,
    kind: 'link',
    to: '/chat',
  },
  {
    id: 'scipPortal',
    title: 'SCIP Portal',
    description:
      'Streamlined Claims Interface Platform for efficient digital submission and processing of administrative claims.',
    icon: FileText,
    kind: 'action',
  },
  {
    id: 'opiContacts',
    title: 'OPI Contacts',
    description:
      "Find FSC & FMC contact information for your unit's financial services and management.",
    icon: Users,
    kind: 'link',
    to: '/opi',
  },
  {
    id: 'resources',
    title: 'Resources',
    description:
      'Access SOPs, how-to guides, FAQs, templates, and comprehensive administrative documentation.',
    icon: Zap,
    kind: 'link',
    to: '/resources',
    badge: 'Under Review',
  },
];

export const footerLinks: LandingFooterLink[] = [
  {
    id: 'about',
    label: 'About',
    icon: Info,
  },
  {
    id: 'contact',
    label: 'Contact',
    icon: Mail,
  },
  {
    id: 'privacy',
    label: 'Privacy',
    icon: ShieldCheck,
  },
];
