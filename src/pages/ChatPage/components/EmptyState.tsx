import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { WELCOME_SUGGESTIONS } from '../constants/suggestions';
import { CategorizedSuggestions } from './CategorizedSuggestions';

interface EmptyStateProps {
  onSuggestionClick: (title: string) => void;
}

// Feature toggle for categorized view - set to true to use new categorized interface
const USE_CATEGORIZED_VIEW = true;

export const EmptyState: React.FC<EmptyStateProps> = ({ onSuggestionClick }) => {
  // If categorized view is enabled, use the new component
  if (USE_CATEGORIZED_VIEW) {
    return (
      <motion.div
        className="flex items-center justify-center h-full p-4 sm:p-6 pb-24 sm:pb-28 overflow-x-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="w-full">
          <CategorizedSuggestions onSuggestionClick={onSuggestionClick} />
        </div>
      </motion.div>
    );
  }

  // Original view (preserved for easy rollback)
  return (
    <motion.div
      className="flex items-center justify-center h-full p-4 sm:p-6 pb-24 sm:pb-28 overflow-x-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center max-w-2xl mx-auto w-full">
        <motion.h2
          className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-8 gradient-text"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          How can I help you today?
        </motion.h2>
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          initial="hidden"
          animate="visible"
          variants={{
            hidden: { opacity: 0 },
            visible: {
              opacity: 1,
              transition: {
                staggerChildren: 0.1,
              },
            },
          }}
        >
          {WELCOME_SUGGESTIONS.map((item, index) => (
            <motion.div
              key={index}
              variants={{
                hidden: { y: 20, opacity: 0 },
                visible: { y: 0, opacity: 1 },
              }}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
            >
              <Card
                className="cursor-pointer group glass rounded-2xl transition-all duration-300 hover:shadow-xl"
                onClick={() => onSuggestionClick(item.title)}
              >
                <CardContent className="p-4 sm:p-6 relative min-h-[80px]">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{item.icon}</span>
                    <div className="flex-1">
                      <div className="font-semibold text-sm sm:text-base mb-1 text-[var(--text)] group-hover:text-[var(--primary)] transition-colors">
                        {item.title}
                      </div>
                      <div className="text-xs sm:text-sm text-[var(--text-secondary)] opacity-80">
                        {item.subtitle}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </motion.div>
  );
};
