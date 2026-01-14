import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Menu,
  X,
  Brain,
  Zap,
  Minimize2,
  HelpCircle,
  Sun,
  Moon,
  Plane,
  Layers,
  FileQuestion,
  ChevronRight,
} from 'lucide-react';
import { 
  Sheet, 
  SheetContent, 
  SheetHeader, 
  SheetTitle, 
  SheetTrigger, 
  Button, 
  Switch, 
  Separator 
} from '@/components/ui';
import { cn } from '@/lib/utils';
import { Link } from 'react-router-dom';

interface HamburgerMenuProps {
  theme: string;
  toggleTheme: () => void;
  modelMode: 'fast' | 'smart';
  setModelMode: (mode: 'fast' | 'smart') => void;
  shortAnswerMode: boolean;
  setShortAnswerMode: (value: boolean) => void;
  onTripPlannerOpen: () => void;
  onHelpOpen: () => void;
  onWhatsNewOpen?: () => void;
  onHowItWorksOpen?: () => void;
  onExportMarkdown: () => void;
  onClearConversation: () => void;
  hasWhatsNew?: boolean;
  isOpen?: boolean;
  onOpenChange?: (value: boolean) => void;
  highlightModelMode?: boolean;
  highlightShortAnswers?: boolean;
}

type ToggleGroupOption = {
  value: string;
  label: string;
  icon: React.ReactNode;
};

interface ToggleGroupMenuItem {
  type: 'toggle-group';
  label: string;
  icon: React.ReactNode;
  value: string;
  options: ToggleGroupOption[];
  onChange: (value: string) => void;
}

interface SwitchMenuItem {
  type: 'switch';
  label: string;
  icon: React.ReactNode;
  value: boolean;
  onChange: (value: boolean) => void;
  description?: string;
}

interface ButtonMenuItem {
  type: 'button';
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
}

interface LinkMenuItem {
  type: 'link';
  label: string;
  icon: React.ReactNode;
  href: string;
}

type MenuItem = ToggleGroupMenuItem | SwitchMenuItem | ButtonMenuItem | LinkMenuItem;

interface MenuSection {
  title: string;
  items: MenuItem[];
}

export const HamburgerMenu: React.FC<HamburgerMenuProps> = ({
  theme,
  toggleTheme,
  modelMode,
  setModelMode,
  shortAnswerMode,
  setShortAnswerMode,
  onTripPlannerOpen,
  onHelpOpen,
  onWhatsNewOpen,
  onHowItWorksOpen,
  onExportMarkdown,
  onClearConversation,
  hasWhatsNew,
  isOpen,
  onOpenChange,
  highlightModelMode = false,
  highlightShortAnswers = false,
}) => {
  const [internalOpen, setInternalOpen] = React.useState(false);
  const controlled = typeof isOpen === 'boolean';
  const open = controlled ? (isOpen as boolean) : internalOpen;
  const setOpen = React.useCallback(
    (value: boolean) => {
      if (!controlled) {
        setInternalOpen(value);
      }
      onOpenChange?.(value);
    },
    [controlled, onOpenChange],
  );

  const modelSectionRef = useRef<HTMLDivElement | null>(null);
  const shortSectionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (open && highlightModelMode && modelSectionRef.current) {
      modelSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [open, highlightModelMode]);

  useEffect(() => {
    if (open && highlightShortAnswers && shortSectionRef.current) {
      shortSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [open, highlightShortAnswers]);

  const toolsItems: MenuItem[] = [
    {
      type: 'button',
      label: 'Travel Planner (Currently under update)',
      icon: <Plane className="w-4 h-4" />,
      onClick: () => {
        setOpen(false);
        onTripPlannerOpen();
      },
    },
    {
      type: 'button',
      label: "What's New",
      icon: <Sun className="w-4 h-4" />,
      onClick: () => {
        setOpen(false);
        onWhatsNewOpen && onWhatsNewOpen();
      },
    },
  ];

  toolsItems.push(
    {
      type: 'button',
      label: 'How this chatbot works',
      icon: <Layers className="w-4 h-4" />,
      onClick: () => {
        setOpen(false);
        onHowItWorksOpen && onHowItWorksOpen();
      },
    },
    {
      type: 'button',
      label: 'Help',
      icon: <HelpCircle className="w-4 h-4" />,
      onClick: () => {
        setOpen(false);
        onHelpOpen();
      },
    },
  );

  const menuSections: MenuSection[] = [
    {
      title: 'AI Settings',
      items: [
        {
          type: 'toggle-group',
          label: 'Model Mode',
          icon: modelMode === 'smart' ? <Brain className="w-4 h-4" /> : <Zap className="w-4 h-4" />,
          value: modelMode,
          options: [
            { value: 'smart', label: 'Smart', icon: <Brain className="w-3 h-3" /> },
            { value: 'fast', label: 'Fast', icon: <Zap className="w-3 h-3" /> },
          ],
          onChange: (value: string) => setModelMode(value as 'fast' | 'smart'),
        },
        {
          type: 'switch',
          label: 'Short Answers',
          description: 'Get concise responses',
          icon: <Minimize2 className="w-4 h-4" />,
          value: shortAnswerMode,
          onChange: setShortAnswerMode,
        },
      ],
    },
    {
      title: 'Tools',
      items: toolsItems,
    },
    {
      title: 'Conversation',
      items: [
        {
          type: 'button',
          label: 'Export as Markdown',
          icon: <FileQuestion className="w-4 h-4" />,
          onClick: onExportMarkdown,
        },
        {
          type: 'button',
          label: 'Clear conversation',
          icon: <X className="w-4 h-4" />,
          onClick: onClearConversation,
        },
      ],
    },
    {
      title: 'Appearance',
      items: [
        {
          type: 'toggle-group',
          label: 'Theme',
          icon: theme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />,
          value: theme,
          options: [
            { value: 'light', label: 'Light', icon: <Sun className="w-3 h-3" /> },
            { value: 'dark', label: 'Dark', icon: <Moon className="w-3 h-3" /> },
          ],
          onChange: (value: string) => {
            if (value !== theme) toggleTheme();
          },
        },
      ],
    },
  ];

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{
            type: 'spring',
            stiffness: 200,
            damping: 20,
            delay: 0.5,
          }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            variant="outline"
            size="icon"
            className="h-11 w-11 rounded-lg border-2 shadow-md hover:shadow-lg transition-all duration-200 bg-[var(--card)] hover:bg-[var(--accent)] hover:border-[var(--accent-foreground)] group relative"
          >
            <Menu className="h-6 w-6 transition-transform duration-200 group-hover:scale-110" />
            {hasWhatsNew && (
              <span className="absolute -top-0.5 -right-0.5 inline-block h-2.5 w-2.5 rounded-full bg-[var(--primary)] ring-2 ring-background" />
            )}
            <span className="sr-only">Open menu</span>
            {/* Pulse animation for new users */}
            <motion.div
              className="absolute inset-0 rounded-lg border-2 border-[var(--accent-foreground)]"
              initial={{ opacity: 0.6, scale: 1 }}
              animate={{
                opacity: [0.6, 0, 0.6],
                scale: [1, 1.15, 1],
              }}
              transition={{
                duration: 2,
                repeat: 2,
                repeatType: 'loop',
                ease: 'easeInOut',
              }}
              style={{ pointerEvents: 'none' }}
            />
          </Button>
        </motion.div>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="flex h-full w-[300px] flex-col overflow-hidden border-l border-[var(--border)] p-0 sm:w-[350px]"
      >
        <SheetHeader className="px-6 py-4 border-b border-[var(--border)]">
          <SheetTitle className="text-xl font-bold">Menu</SheetTitle>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto overscroll-contain pb-[max(env(safe-area-inset-bottom),1rem)]">
          {menuSections.map((section, sectionIndex) => (
            <div key={section.title} className="px-6 py-4">
              <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-3">
                {section.title}
              </h3>
              <div className="space-y-2">
                {section.items.map((item, itemIndex) => {
                  if (item.type === 'toggle-group') {
                    const isModelToggle = item.label === 'Model Mode';
                    const highlight = isModelToggle && highlightModelMode;
                    const glowColour = 'rgba(59,130,246,0.45)';
                    return (
                      <motion.div
                        key={itemIndex}
                        ref={isModelToggle ? modelSectionRef : undefined}
                        className={cn('space-y-2 rounded-lg', highlight && 'bg-[var(--primary)]/5')}
                        animate={
                          highlight
                            ? {
                                boxShadow: [
                                  '0 0 0 0 rgba(0,0,0,0)',
                                  `0 0 0 12px ${glowColour}`,
                                  '0 0 0 0 rgba(0,0,0,0)',
                                ],
                                scale: [1, 1.03, 1],
                              }
                            : {
                                boxShadow: '0 0 0 0 rgba(0,0,0,0)',
                                scale: 1,
                              }
                        }
                        transition={highlight ? { duration: 1.1, repeat: 1 } : undefined}
                      >
                        <div className="flex items-center gap-2 text-sm font-medium">
                          {item.icon}
                          <span>{item.label}</span>
                        </div>
                        <div className="flex gap-1 p-1 bg-[var(--background-secondary)] rounded-lg">
                          {item.options.map((option) => (
                            <button
                              key={option.value}
                              onClick={() => item.onChange(option.value)}
                              className={cn(
                                'flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                                item.value === option.value
                                  ? theme === 'dark'
                                    ? 'bg-yellow-500 text-black shadow-sm'
                                    : 'bg-green-500 text-white shadow-sm'
                                  : 'text-[var(--text-secondary)] hover:text-foreground',
                              )}
                            >
                              {option.icon}
                              {option.label}
                            </button>
                          ))}
                        </div>
                      </motion.div>
                    );
                  }

                  if (item.type === 'switch') {
                    const highlight = item.label === 'Short Answers' && highlightShortAnswers;
                    const glowColour = 'rgba(217,119,6,0.55)';
                    return (
                      <motion.div
                        key={itemIndex}
                        ref={item.label === 'Short Answers' ? shortSectionRef : undefined}
                        className={cn(
                          'flex items-center justify-between py-2 rounded-lg px-2',
                          highlight && 'bg-amber-500/10',
                        )}
                        animate={
                          highlight
                            ? {
                                boxShadow: [
                                  '0 0 0 0 rgba(0,0,0,0)',
                                  `0 0 0 12px ${glowColour}`,
                                  '0 0 0 0 rgba(0,0,0,0)',
                                ],
                                scale: [1, 1.04, 1],
                              }
                            : {
                                boxShadow: '0 0 0 0 rgba(0,0,0,0)',
                                scale: 1,
                              }
                        }
                        transition={highlight ? { duration: 1.1, repeat: 1 } : undefined}
                      >
                        <div className="flex items-center gap-3">
                          {item.icon}
                          <div>
                            <div className="text-sm font-medium">{item.label}</div>
                            {item.description && (
                              <div className="text-xs text-[var(--text-secondary)]">
                                {item.description}
                              </div>
                            )}
                          </div>
                        </div>
                        <Switch checked={item.value} onCheckedChange={item.onChange} />
                      </motion.div>
                    );
                  }

                  if (item.type === 'button') {
                    return (
                      <motion.button
                        key={itemIndex}
                        onClick={item.onClick}
                        className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-[var(--background-secondary)] transition-colors text-left"
                        whileHover={{ x: 2 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <div className="flex items-center gap-3">
                          {item.icon}
                          <span className="text-sm font-medium">{item.label}</span>
                        </div>
                        <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
                      </motion.button>
                    );
                  }

                  if (item.type === 'link') {
                    return (
                      <Link key={itemIndex} to={item.href || '/'} onClick={() => setOpen(false)}>
                        <motion.div
                          className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-[var(--background-secondary)] transition-colors"
                          whileHover={{ x: 2 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <div className="flex items-center gap-3">
                            {item.icon}
                            <span className="text-sm font-medium">{item.label}</span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
                        </motion.div>
                      </Link>
                    );
                  }

                  return null;
                })}
              </div>
              {sectionIndex < menuSections.length - 1 && <Separator className="mt-4" />}
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  );
};
