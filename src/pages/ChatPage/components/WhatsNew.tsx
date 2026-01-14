import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Calendar, ChevronDown } from 'lucide-react';
import { WHATS_NEW_BY_DATE } from '../constants/whatsNew';

interface WhatsNewProps {
  className?: string;
}

export const WhatsNew: React.FC<WhatsNewProps> = ({ className = '' }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);

  const updatesByDate = WHATS_NEW_BY_DATE;

  return (
    <motion.div
      className={`w-full max-w-2xl mx-auto mb-8 ${className}`}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="glass border border-[var(--border)] backdrop-blur-xl">
        <CardContent className="p-4">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="flex items-center justify-between w-full mb-3 group hover:opacity-80 transition-opacity"
          >
            <h3 className="text-lg font-semibold text-[var(--text)] flex items-center gap-2">
              <Calendar className="w-5 h-5 text-[var(--primary)] flex-shrink-0" />
              <span>What's New</span>
            </h3>
            <motion.div
              animate={{ rotate: isCollapsed ? 0 : 180 }}
              transition={{ duration: 0.2 }}
              className="flex-shrink-0"
            >
              <ChevronDown className="w-5 h-5 text-[var(--text-secondary)]" />
            </motion.div>
          </button>

          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="space-y-6">
                  {updatesByDate.map((dateGroup, groupIndex) => (
                    <div key={groupIndex}>
                      <p className="text-sm text-[var(--text-secondary)] mb-3 font-medium">
                        {dateGroup.date}
                      </p>
                      <ul className="space-y-3">
                        {dateGroup.updates.map((update, updateIndex) => (
                          <motion.li
                            key={`${groupIndex}-${updateIndex}`}
                            className="flex items-start gap-3 text-sm"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{
                              delay:
                                0.1 * (groupIndex * dateGroup.updates.length + updateIndex + 1),
                            }}
                          >
                            <span className="text-[var(--primary)] mt-0.5 flex-shrink-0">
                              {update.icon}
                            </span>
                            <div className="flex-1 text-left">
                              <div className="text-[var(--text)]">{update.text}</div>
                              {update.description && (
                                <div className="text-xs text-[var(--text-secondary)] mt-1">
                                  {update.description}
                                </div>
                              )}
                            </div>
                          </motion.li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};
