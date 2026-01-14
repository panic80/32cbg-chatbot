import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import {
  HelpCircle,
  Plane,
  FileText,
  DollarSign,
  MessageCircle,
  AlertTriangle,
  Copy,
  Check,
  BookOpen,
  Shield,
  Brain,
  Lightbulb,
  CornerDownLeft,
} from 'lucide-react';

interface HelpDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onInsertExample?: (text: string) => void;
}

export const HelpDialog: React.FC<HelpDialogProps> = ({ open, onOpenChange, onInsertExample }) => {
  const [copiedExample, setCopiedExample] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<string | null>(null);

  const handleSelectExample = async (text: string) => {
    if (onInsertExample) {
      try {
        setIsLoading(text);
        onInsertExample(text);
        onOpenChange(false);
      } finally {
        setIsLoading(null);
      }
    } else {
      try {
        setIsLoading(text);
        await navigator.clipboard.writeText(text);
        setCopiedExample(text);
        setTimeout(() => setCopiedExample(null), 2000);
      } catch (error) {
        console.error('Failed to copy text:', error);
      } finally {
        setIsLoading(null);
      }
    }
  };

  const knowledgeBase = [
    {
      icon: <Plane className="w-5 h-5 text-blue-500" />,
      title: 'CFTDTI',
      description: 'Canadian Forces Temporary Duty Travel Instructions',
    },
    {
      icon: <FileText className="w-5 h-5 text-green-500" />,
      title: 'NJC Travel',
      description: 'National Joint Council Travel Directive',
    },
    {
      icon: <DollarSign className="w-5 h-5 text-purple-500" />,
      title: 'CBI',
      description: 'Compensation and Benefits Instructions',
    },
  ];

  const simpleExample2 = 'What is the private non-commercial accommodation (PNCA) rate?';
  const simpleExample3 = 'What is the current incidental allowance amount?';

  const complexReasoningExample =
    'Determine my TD entitlements for a 1.5-day trip using a personal vehicle: depart 10:00 Day 1, return 15:30 Day 2. List eligible meals, incidental, and mileage with CFTDTI/NJC references.';
  const complexReasoningExample2 =
    'For a same-day TD departing 05:30 and returning 22:15 by personal vehicle, list claimable meals, incidental eligibility, and mileage with CFTDTI/NJC references.';
  const complexReasoningExample3 =
    'For a 3-day TD where lunch is provided on Day 2 and lodging is PNCA each night, calculate daily entitlements and required deductions with references.';

  const exampleCategories = [
    {
      type: 'simple',
      icon: <MessageCircle className="w-4 h-4 text-[var(--text-secondary)]" />,
      title: 'Simple Lookups',
      description: 'Quick rates, definitions, who-to-contact',
      examples: ['What is the current meal rate in Ontario?', simpleExample2, simpleExample3],
    },
    {
      type: 'complex',
      icon: <Brain className="w-4 h-4 text-[var(--text-secondary)]" />,
      title: 'Complex Reasoning',
      description: 'Scenarios combining rules, time windows, and multiple directives',
      examples: [complexReasoningExample, complexReasoningExample2, complexReasoningExample3],
    },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-4xl w-[95vw] sm:w-[90vw] max-h-[90vh] overflow-y-auto glass border-[var(--border)] bg-[var(--card)] focus:outline-none"
        role="dialog"
        aria-labelledby="help-dialog-title"
        aria-describedby="help-dialog-description"
      >
        <DialogHeader className="sticky top-0 bg-[var(--card)] z-10 pb-4 border-b border-[var(--border)]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[var(--primary)]/10">
              <HelpCircle className="w-6 h-6 text-[var(--primary)]" />
            </div>
            <div>
              <DialogTitle
                id="help-dialog-title"
                className="text-2xl font-bold text-[var(--text)] text-left"
              >
                Policy Assistant Help
              </DialogTitle>
              <DialogDescription
                id="help-dialog-description"
                className="text-[var(--text-secondary)] text-left"
              >
                Learn how to use the assistant effectively
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <motion.div
          className="space-y-6 text-[var(--text)] pt-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Examples */}
          <Card className="border-[var(--border)] hover:border-[var(--primary)]/30 transition-colors">
            <CardContent className="p-5">
              <h3 className="text-lg font-semibold mb-3 text-[var(--text)] flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-[var(--primary)]" />
                Examples
              </h3>
              <p className="text-xs text-[var(--text-secondary)] mb-4">
                Click any example to {onInsertExample ? 'insert it' : 'copy it'}.
              </p>

              <div className="space-y-4">
                {exampleCategories.map((category) => (
                  <div key={category.type}>
                    <div className="flex items-center gap-2 mb-3">
                      {category.icon}
                      <span className="font-medium text-sm text-[var(--text)]">
                        {category.title}
                      </span>
                      <span className="text-xs text-[var(--text-secondary)]">
                        • {category.description}
                      </span>
                    </div>
                    <div className="space-y-2 ml-6">
                      {category.examples.map((example) => (
                        <div
                          key={example}
                          className="group flex items-start justify-between p-3 rounded-lg hover:bg-[var(--background-secondary)] cursor-pointer transition-colors border border-transparent hover:border-[var(--border)]"
                          onClick={() => handleSelectExample(example)}
                        >
                          <div className="flex items-start gap-3 flex-1 min-w-0">
                            <div className="text-sm text-[var(--text-secondary)] break-words">
                              "{example}"
                            </div>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="opacity-0 group-hover:opacity-100 transition-opacity h-6 px-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectExample(example);
                            }}
                            disabled={isLoading === example}
                          >
                            {isLoading === example ? (
                              <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
                            ) : onInsertExample ? (
                              <span className="flex items-center gap-1 text-xs">
                                <CornerDownLeft className="w-3 h-3" /> Insert
                              </span>
                            ) : copiedExample === example ? (
                              <Check className="w-3 h-3 text-green-500" />
                            ) : (
                              <Copy className="w-3 h-3" />
                            )}
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Start */}
          <Card className="border-[var(--border)] hover:border-[var(--primary)]/30 transition-colors">
            <CardContent className="p-5">
              <h3 className="text-lg font-semibold mb-4 text-[var(--text)] flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-[var(--primary)]" />
                Quick Start Guide
              </h3>
              <div className="grid gap-3 text-sm">
                <div className="flex items-start gap-3 p-3 rounded-lg bg-[var(--background-secondary)]/50">
                  <span className="flex-shrink-0 w-6 h-6 bg-[var(--primary)] text-white rounded-full flex items-center justify-center text-xs font-bold">
                    1
                  </span>
                  <div>
                    <div className="font-medium text-[var(--text)]">Be Specific</div>
                    <div className="text-[var(--text-secondary)]">
                      Include times, locations, or policy names in your questions
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 rounded-lg bg-[var(--background-secondary)]/50">
                  <span className="flex-shrink-0 w-6 h-6 bg-[var(--primary)] text-white rounded-full flex items-center justify-center text-xs font-bold">
                    2
                  </span>
                  <div>
                    <div className="font-medium text-[var(--text)]">Use Follow-ups</div>
                    <div className="text-[var(--text-secondary)]">
                      Ask for sources, clarification, or additional details
                    </div>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 rounded-lg bg-[var(--background-secondary)]/50">
                  <span className="flex-shrink-0 w-6 h-6 bg-[var(--primary)] text-white rounded-full flex items-center justify-center text-xs font-bold">
                    3
                  </span>
                  <div>
                    <div className="font-medium text-[var(--text)]">Try Examples</div>
                    <div className="text-[var(--text-secondary)]">
                      Click the examples above to get started quickly
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Sources & Limits */}
          <Card className="border-[var(--border)] hover:border-[var(--primary)]/30 transition-colors">
            <CardContent className="p-5">
              <h3 className="text-lg font-semibold mb-4 text-[var(--text)] flex items-center gap-2">
                <Shield className="w-5 h-5 text-[var(--primary)]" />
                Sources & Limits
              </h3>
              <div className="space-y-3">
                {knowledgeBase.map((item, index) => (
                  <div
                    key={index}
                    className="flex items-start gap-3 p-3 rounded-lg bg-[var(--background-secondary)] hover:bg-[var(--background-secondary)]/80 transition-colors"
                  >
                    {item.icon}
                    <div>
                      <div className="font-medium text-[var(--text)]">{item.title}</div>
                      <div className="text-sm text-[var(--text-secondary)]">{item.description}</div>
                    </div>
                  </div>
                ))}
                <div className="flex items-start gap-3 p-3 rounded-lg border border-orange-500/20 bg-orange-50/5">
                  <AlertTriangle className="w-5 h-5 text-orange-500 mt-0.5" />
                  <div className="text-sm text-[var(--text-secondary)]">
                    Always verify critical information with official sources or your unit's
                    administrative staff. This assistant provides guidance and references but does
                    not replace official policy documents.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          {/* Tips removed; essentials covered in Quick Start to streamline */}
        </motion.div>
      </DialogContent>
    </Dialog>
  );
};
