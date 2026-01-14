import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';
import { MAINTENANCE_MESSAGE } from '@/constants';

export const MaintenanceBanner: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="w-full bg-amber-500/10 dark:bg-amber-500/20 border-b border-amber-500/30 dark:border-amber-400/30"
    >
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-center gap-3">
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        >
          <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0" />
        </motion.div>
        <p className="text-sm font-medium text-amber-900 dark:text-amber-100 text-center">
          {MAINTENANCE_MESSAGE}
        </p>
      </div>
    </motion.div>
  );
};
