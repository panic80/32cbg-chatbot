import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem, setLocalStorageItem } from '@/utils/storage';
import { CATEGORIZED_SUGGESTIONS, type SuggestionItem } from '../constants/suggestions';

// Security: Styles moved to external CSS file to avoid inline HTML injection risks
import './CategorizedSuggestions.css';

interface CategorizedSuggestionsProps {
  onSuggestionClick: (title: string) => void;
}

export const CategorizedSuggestions: React.FC<CategorizedSuggestionsProps> = ({
  onSuggestionClick,
}) => {
  const [activeTab, setActiveTab] = useState(CATEGORIZED_SUGGESTIONS[0]?.id || '');

  // A/B variant selection for chat tabs (persisted in localStorage)
  // Validates input to prevent injection via URL params
  const variant = useMemo(() => {
    if (typeof window === 'undefined') return 'A';
    const params = new URLSearchParams(window.location.search);
    const qp = (params.get('tabsVariant') || '').toUpperCase();
    // Security: Validate variant is exactly 'A' or 'B' to prevent injection
    let v = qp || (getLocalStorageItem(StorageKeys.tabsVariant) || '').toUpperCase();
    if (v !== 'A' && v !== 'B') v = Math.random() < 0.5 ? 'A' : 'B';
    setLocalStorageItem(StorageKeys.tabsVariant, v);
    return v as 'A' | 'B';
  }, []);

  const renderQuestionGrid = (questions: SuggestionItem[]) => (
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
      {questions.map((item, index) => (
        <motion.div
          key={`${item.title}-${index}`}
          variants={{
            hidden: { y: 20, opacity: 0 },
            visible: { y: 0, opacity: 1 },
          }}
          whileHover={{ scale: 1.02, y: -2 }}
          whileTap={{ scale: 0.98 }}
        >
          <Card
            className="cursor-pointer group glass rounded-2xl transition-all duration-300 hover:shadow-xl h-full"
            onClick={() => onSuggestionClick(item.title)}
          >
            <CardContent className="p-4 sm:p-6 relative h-full flex items-center">
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
  );

  return (
    <div className={`w-full max-w-4xl mx-auto ${variant === 'B' ? 'variant-b' : 'variant-a'}`}>
      {/* Styles loaded from external CategorizedSuggestions.css file for XSS protection */}
      <motion.h2
        className="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8 gradient-text text-center"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        How can I help you today?
      </motion.h2>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full mb-6 h-auto">
            {CATEGORIZED_SUGGESTIONS.map((category) => (
              <TabsTrigger
                key={category.id}
                value={category.id}
                className="flex flex-col items-center gap-1 p-3 text-xs sm:text-sm rounded-md transition-all duration-200"
              >
                <span className="text-lg">{category.icon}</span>
                <span className="hidden sm:inline">{category.label}</span>
                <span className="sm:hidden">
                  {category.shortLabel ?? category.label.split(' ')[0]}
                </span>
              </TabsTrigger>
            ))}
          </TabsList>

          {CATEGORIZED_SUGGESTIONS.map((category) => (
            <TabsContent key={category.id} value={category.id} className="mt-0">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                {renderQuestionGrid(category.questions)}
              </motion.div>
            </TabsContent>
          ))}
        </Tabs>
      </motion.div>
    </div>
  );
};
