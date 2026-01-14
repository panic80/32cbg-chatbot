/**
 * Smart Action Chip Component
 * Mobile-optimized chip component for suggested questions
 */

import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { FollowUpQuestion } from '@/types';
import { cn } from '@/lib/utils';
import { getSuggestionCategoryIcon } from '@/utils/suggestionCategories';

interface SmartActionChipProps {
  question: FollowUpQuestion;
  onClick: (question: string) => void;
  index: number;
  className?: string;
}

const getCategoryColor = (category?: string) => {
  switch (category) {
    case 'clarification':
      return 'border-blue-200 hover:border-blue-400 hover:bg-blue-50/80 dark:border-blue-700 dark:hover:border-blue-500 dark:hover:bg-blue-900/30';
    case 'related':
      return 'border-green-200 hover:border-green-400 hover:bg-green-50/80 dark:border-green-700 dark:hover:border-green-500 dark:hover:bg-green-900/30';
    case 'practical':
      return 'border-orange-200 hover:border-orange-400 hover:bg-orange-50/80 dark:border-orange-700 dark:hover:border-orange-500 dark:hover:bg-orange-900/30';
    case 'explore':
      return 'border-purple-200 hover:border-purple-400 hover:bg-purple-50/80 dark:border-purple-700 dark:hover:border-purple-500 dark:hover:bg-purple-900/30';
    default:
      return 'border-gray-200 hover:border-gray-400 hover:bg-gray-50/80 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-800/30';
  }
};

const SmartActionChip: React.FC<SmartActionChipProps> = ({
  question,
  onClick,
  index,
  className = '',
}) => {
  // Ensure we have a proper question object and extract the question text safely
  const questionText = React.useMemo(() => {
    if (typeof question === 'string') {
      return question;
    }
    if (question && typeof question === 'object' && 'question' in question) {
      return typeof question.question === 'string' ? question.question : 'Question not available';
    }
    return 'Question not available';
  }, [question]);

  // Safety check
  if (!questionText) {
    return null;
  }

  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.8, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{
        delay: index * 0.1,
        type: 'spring',
        stiffness: 300,
        damping: 20,
      }}
      whileHover={{
        scale: 1.02,
        y: -2,
        transition: { type: 'spring', stiffness: 400, damping: 25 },
      }}
      whileTap={{ scale: 0.96 }}
      onClick={() => onClick(questionText)}
      className={cn(
        'group relative flex items-start gap-2 px-3 py-2 rounded-lg border bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm shadow-sm transition-all duration-200',
        'min-h-[36px] text-left text-sm font-medium text-gray-700 dark:text-gray-200',
        'hover:bg-white/90 dark:hover:bg-gray-800/90 hover:shadow-md active:shadow-sm',
        'focus:outline-none focus:ring-1 focus:ring-blue-500 dark:focus:ring-blue-400 focus:ring-offset-1',
        'max-w-full',
        getCategoryColor(question?.category)
          .replace('border-', 'border-')
          .replace('hover:border-', 'hover:border-')
          .replace('hover:bg-', 'hover:bg-'),
        className,
      )}
      title={questionText} // Full question on hover
    >
      {/* Category icon */}
      <motion.div
        className="flex-shrink-0 opacity-70 group-hover:opacity-100 transition-opacity mt-0.5"
        whileHover={{ rotate: 15 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        {getSuggestionCategoryIcon(question?.category, { variant: 'chip', size: 12 })}
      </motion.div>

      {/* Question text */}
      <span className="flex-1 text-left whitespace-normal break-words">{questionText}</span>

      {/* Confidence indicators */}
      <div className="flex items-center gap-1 flex-shrink-0">
        {question?.confidence && question.confidence > 0.8 && (
          <motion.div
            className="w-2 h-2 rounded-full bg-green-500 opacity-60"
            title="High confidence question"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, delay: 0.2 + index * 0.1 }}
          />
        )}
        {question?.groundingScore && question.groundingScore > 0.5 && (
          <motion.div
            className="w-2 h-2 rounded-full bg-blue-500 opacity-60"
            title="Well-grounded in source content"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 500, delay: 0.3 + index * 0.1 }}
          />
        )}
      </div>

      {/* Hover arrow indicator */}
      <motion.div
        className="opacity-0 group-hover:opacity-100 transition-opacity"
        initial={{ x: -5 }}
        whileHover={{ x: 0 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        <ArrowRight size={12} className="text-gray-500 dark:text-gray-400" />
      </motion.div>

      {/* Subtle gradient overlay on hover */}
      <motion.div
        className="absolute inset-0 rounded-full bg-gradient-to-r from-transparent via-white to-transparent opacity-0 pointer-events-none"
        whileHover={{ opacity: 0.1 }}
        transition={{ duration: 0.3 }}
      />
    </motion.button>
  );
};

export default SmartActionChip;
