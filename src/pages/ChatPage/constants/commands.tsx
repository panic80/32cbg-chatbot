import { Zap, Hash, AtSign, HelpCircle } from 'lucide-react';

export const INLINE_COMMANDS = [
  {
    icon: <Zap size={16} />,
    label: 'Quick response',
    command: '/quick',
    description: 'Get a concise answer',
  },
  {
    icon: <Hash size={16} />,
    label: 'Summarize',
    command: '/summarize',
    description: 'Summarize the conversation',
  },
  {
    icon: <AtSign size={16} />,
    label: 'Mention policy',
    command: '/policy',
    description: 'Reference specific policy',
  },
  {
    icon: <HelpCircle size={14} />,
    label: 'Explain',
    command: '/explain',
    description: 'Get detailed explanation',
  },
];
