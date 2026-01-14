import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { HelpCircle, ArrowRight } from 'lucide-react';
import { FollowUpQuestion } from '@/types/chat';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getSuggestionCategoryIcon,
  getSuggestionCategoryLabel,
} from '@/utils/suggestionCategories';

interface FollowUpQuestionsProps {
  questions: FollowUpQuestion[];
  onQuestionClick: (question: string) => void;
  className?: string;
}

const FollowUpQuestions: React.FC<FollowUpQuestionsProps> = ({
  questions,
  onQuestionClick,
  className = '',
}) => {
  if (!questions || questions.length === 0) {
    return null;
  }

  return (
    <motion.div
      className={cn('mt-4 space-y-2', className)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <motion.div
        className="flex items-center gap-2 text-sm text-[var(--text-secondary)] font-medium"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1 }}
      >
        <HelpCircle size={16} className="opacity-70" />
        <span>Suggested questions</span>
      </motion.div>

      <motion.div
        className="space-y-2"
        initial="hidden"
        animate="visible"
        variants={{
          hidden: { opacity: 0 },
          visible: {
            opacity: 1,
            transition: {
              staggerChildren: 0.1,
              delayChildren: 0.2,
            },
          },
        }}
      >
        {questions.map((question) => (
          <motion.div
            key={question.id}
            variants={{
              hidden: { opacity: 0, x: -20 },
              visible: { opacity: 1, x: 0 },
            }}
            whileHover={{ x: 4 }}
            whileTap={{ scale: 0.98 }}
          >
            <Card
              className="cursor-pointer transition-all duration-200 hover:shadow-md hover:border-[var(--primary)] group border border-[var(--border)] bg-[var(--card)] hover:bg-[var(--background-secondary)] overflow-hidden relative"
              onClick={() => onQuestionClick(question.question)}
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-[var(--primary)] to-transparent opacity-0"
                initial={false}
                whileHover={{ opacity: 0.05 }}
                transition={{ duration: 0.3 }}
              />

              <CardContent className="p-3 relative">
                <div className="flex items-start gap-3">
                  <motion.div
                    className="mt-1 opacity-70 group-hover:opacity-100 transition-opacity"
                    whileHover={{ rotate: 15 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    {getSuggestionCategoryIcon(question.category, { variant: 'card', size: 14 })}
                  </motion.div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-[var(--text-secondary)] opacity-80">
                        {getSuggestionCategoryLabel(question.category)}
                      </span>
                      <AnimatePresence>
                        {question.confidence && question.confidence > 0.8 && (
                          <motion.div
                            className="w-2 h-2 rounded-full bg-green-500 opacity-60"
                            title="High confidence question"
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            exit={{ scale: 0 }}
                            transition={{ type: 'spring', stiffness: 500 }}
                          />
                        )}
                        {question.groundingScore && question.groundingScore > 0.5 && (
                          <motion.div
                            className="w-2 h-2 rounded-full bg-blue-500 opacity-60"
                            title="Well-grounded in source content"
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            exit={{ scale: 0 }}
                            transition={{ type: 'spring', stiffness: 500, delay: 0.1 }}
                          />
                        )}
                      </AnimatePresence>
                    </div>

                    <p className="text-sm text-[var(--text)] leading-relaxed group-hover:text-[var(--primary)] transition-colors">
                      {question.question}
                    </p>

                    {question.sourceGrounding && (
                      <motion.p
                        className="text-xs text-[var(--text-secondary)] opacity-60 mt-1 italic"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 0.6, height: 'auto' }}
                        transition={{ delay: 0.2 }}
                      >
                        Based on: {question.sourceGrounding}
                      </motion.p>
                    )}
                  </div>

                  <motion.div
                    className="opacity-0 group-hover:opacity-100 transition-opacity mt-1"
                    initial={{ x: -10 }}
                    whileHover={{ x: 0 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <ArrowRight size={14} className="text-[var(--text-secondary)]" />
                  </motion.div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        className="text-xs text-[var(--text-secondary)]/60 italic mt-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        transition={{ delay: 0.5 }}
      >
        Click on any question to continue the conversation
      </motion.div>
    </motion.div>
  );
};

export default FollowUpQuestions;
