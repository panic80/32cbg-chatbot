/**
 * Enhanced Suggestion Controller - Phased Interaction Model
 * Implements smart progressive disclosure for follow-up questions
 */

import React, { useReducer, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { FollowUpQuestion } from '@/types/chat';
import { cn } from '@/lib/utils';
import FollowUpQuestions from './FollowUpQuestions';
import SmartActionChip from './SmartActionChip';

interface SuggestionState {
  phase: 'hidden' | 'smart' | 'full';
  isVisible: boolean;
  timerId: number | null;
}

type SuggestionAction =
  | { type: 'VIEWPORT_ENTER' }
  | { type: 'VIEWPORT_EXIT' }
  | { type: 'SHOW_SMART_ACTIONS' }
  | { type: 'SHOW_ALL' }
  | { type: 'HIDE_ALL' }
  | { type: 'CANCEL_TIMER' };

const initialState: SuggestionState = {
  phase: 'smart', // Start with smart actions visible immediately
  isVisible: true,
  timerId: null,
};

function suggestionReducer(state: SuggestionState, action: SuggestionAction): SuggestionState {
  switch (action.type) {
    case 'VIEWPORT_ENTER':
      return { ...state, isVisible: true };

    case 'VIEWPORT_EXIT':
      return { ...state, isVisible: false, timerId: null };

    case 'SHOW_SMART_ACTIONS':
      return { ...state, phase: 'smart', timerId: null };

    case 'SHOW_ALL':
      return { ...state, phase: 'full' };

    case 'HIDE_ALL':
      return { ...state, phase: 'hidden' };

    case 'CANCEL_TIMER':
      return { ...state, timerId: null };

    default:
      return state;
  }
}

interface SuggestionControllerProps {
  questions: FollowUpQuestion[];
  onQuestionClick: (question: string) => void;
  messageId: string;
  isLatestMessage?: boolean;
  className?: string;
}

// Smart action selection algorithm with confidence + diversity scoring
function selectSmartActions(
  suggestions: FollowUpQuestion[],
  maxCount = 3,
  isLatestMessage = false,
): FollowUpQuestion[] {
  if (!suggestions || suggestions.length === 0) return [];
  if (suggestions.length <= maxCount) return suggestions;

  // Dynamic diversity weighting based on suggestion count
  const DIVERSITY_THRESHOLD = 8;
  const maxDiversityWeight = 0.3;
  const diversityWeight =
    maxDiversityWeight * (1 / (1 + Math.exp(-(suggestions.length - DIVERSITY_THRESHOLD) / 3)));

  // Filter for high-quality candidates
  const candidates = suggestions.filter(
    (s) => (s.confidence || 0) > 0.6 && (s.groundingScore || 0) > 0.3,
  );

  if (candidates.length === 0) {
    // Fallback to top suggestions by confidence
    return [...suggestions]
      .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
      .slice(0, maxCount);
  }

  const selection: FollowUpQuestion[] = [];
  const usedCategories = new Set<string>();

  // Sort by confidence
  const sortedCandidates = [...candidates].sort(
    (a, b) => (b.confidence || 0) - (a.confidence || 0),
  );

  // Always pick the highest confidence one first
  const topSuggestion = sortedCandidates[0];
  if (topSuggestion) {
    selection.push(topSuggestion);
    if (topSuggestion.category) {
      usedCategories.add(topSuggestion.category);
    }
  }

  // Score remaining candidates
  const remaining = sortedCandidates.slice(1);
  while (selection.length < maxCount && remaining.length > 0) {
    const scored = remaining.map((s) => {
      const baseScore = s.confidence || 0;
      const diversityBonus = s.category && !usedCategories.has(s.category) ? diversityWeight : 0;
      const recencyBonus = isLatestMessage ? 0.1 : 0; // Small boost for latest message

      return {
        ...s,
        score: baseScore + diversityBonus + recencyBonus,
      };
    });

    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];

    if (best) {
      selection.push(best);
      if (best.category) {
        usedCategories.add(best.category);
      }

      // Remove selected item from remaining
      const index = remaining.findIndex((s) => s.id === best.id);
      if (index > -1) {
        remaining.splice(index, 1);
      }
    } else {
      break;
    }
  }

  return selection;
}

const SuggestionController: React.FC<SuggestionControllerProps> = ({
  questions,
  onQuestionClick,
  messageId,
  isLatestMessage = false,
  className = '',
}) => {
  const [state, dispatch] = useReducer(suggestionReducer, initialState);
  const observerRef = useRef<HTMLDivElement>(null);

  // Smart actions selection
  const smartActions = useMemo(
    () => selectSmartActions(questions, 3, isLatestMessage),
    [questions, isLatestMessage],
  );

  if (!questions || questions.length === 0) {
    return null;
  }

  return (
    <div
      ref={observerRef}
      className={cn('mt-4 mb-4', className)}
      style={{ minHeight: state.phase !== 'hidden' ? '40px' : '0px' }}
    >
      {/* ARIA live region for accessibility */}
      <div aria-live="polite" className="sr-only">
        {state.phase === 'smart' && `${smartActions.length} suggested questions available`}
        {state.phase === 'full' && `${questions.length} questions available`}
      </div>

      <AnimatePresence mode="wait">
        {state.phase === 'smart' && (
          <motion.div
            key="smart-actions"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="space-y-3 p-3 rounded-lg bg-gray-50/50 dark:bg-gray-800/30 border border-gray-200/50 dark:border-gray-700/50"
          >
            {/* Smart action chips - subtle and non-protruding */}
            {/* Suggested follow-up questions label */}
            <div className="flex items-center gap-2 mb-2">
              <motion.div
                whileHover={{ rotate: 15 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <HelpCircle size={14} className="text-gray-500 dark:text-gray-400" />
              </motion.div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                Suggested follow-up questions
              </span>
            </div>
            <div className="flex flex-wrap gap-2 opacity-75 hover:opacity-100 transition-opacity duration-300">
              {smartActions.map((question, index) => (
                <SmartActionChip
                  key={question.id}
                  question={question}
                  onClick={onQuestionClick}
                  index={index}
                  className=""
                />
              ))}
            </div>

            {/* Show more button */}
            {questions.length > smartActions.length && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => dispatch({ type: 'SHOW_ALL' })}
                  className="text-[var(--text-secondary)] hover:text-[var(--primary)] text-xs opacity-60 hover:opacity-100"
                >
                  <ChevronDown size={12} className="mr-1" />
                  {questions.length - smartActions.length} more
                </Button>
              </motion.div>
            )}
          </motion.div>
        )}

        {state.phase === 'full' && (
          <motion.div
            key="full-view"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
          >
            <FollowUpQuestions
              questions={questions}
              onQuestionClick={onQuestionClick}
              className="mt-0"
            />

            {/* Collapse button */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="mt-3"
            >
              <Button
                variant="ghost"
                size="sm"
                onClick={() => dispatch({ type: 'SHOW_SMART_ACTIONS' })}
                className="text-[var(--text-secondary)] hover:text-[var(--primary)] text-xs"
              >
                <ChevronDown size={14} className="mr-1 rotate-180" />
                Show less
              </Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SuggestionController;
