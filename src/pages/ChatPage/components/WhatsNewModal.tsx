import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Calendar } from 'lucide-react';
import { motion } from 'framer-motion';
import { StorageKeys } from '@/constants/storage';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { WHATS_NEW_BY_DATE, WHATS_NEW_VERSION } from '../constants/whatsNew';

interface WhatsNewModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const WhatsNewModal: React.FC<WhatsNewModalProps> = ({ open, onOpenChange }) => {
  const [lastSeenVersion, setLastSeenVersion] = useLocalStorage<string>(
    StorageKeys.whatsNewLastSeen,
    '',
  );

  const markSeen = () => setLastSeenVersion(WHATS_NEW_VERSION);

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      markSeen();
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>What’s New</DialogTitle>
          <DialogDescription>
            Release notes and recent improvements. Version {WHATS_NEW_VERSION}
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className="max-h-[60vh] pr-2">
          <div className="space-y-6 pt-2">
            {WHATS_NEW_BY_DATE.map((dateGroup, groupIndex) => (
              <div key={groupIndex}>
                <div className="flex items-center gap-2 mb-3 text-[var(--text-secondary)] text-sm font-medium">
                  <Calendar className="w-4 h-4" />
                  <span>{dateGroup.date}</span>
                </div>
                <ul className="space-y-3 pl-0">
                  {dateGroup.updates.map((update, idx) => (
                    <motion.li
                      key={`${groupIndex}-${idx}`}
                      className="flex items-start gap-3 text-sm"
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.02 * (idx + 1) }}
                    >
                      <span className="text-[var(--primary)] mt-0.5 flex-shrink-0">
                        {update.icon}
                      </span>
                      <div className="flex-1">
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
        </ScrollArea>
        <div className="flex items-center justify-end pt-2 gap-2">
          <Button size="sm" onClick={() => handleOpenChange(false)}>
            Got it
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WhatsNewModal;
