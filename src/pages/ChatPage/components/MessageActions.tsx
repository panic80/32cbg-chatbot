import React from 'react';
import { motion } from 'framer-motion';
import { Copy, RefreshCw, Volume2, ThumbsUp, ThumbsDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';

interface MessageActionsProps {
  onCopy: () => void;
  onRegenerate: () => void;
  onVoice: () => void;
  isLoading: boolean;
  onFeedback?: (value: 'up' | 'down') => void;
}

export const MessageActions: React.FC<MessageActionsProps> = ({
  onCopy,
  onRegenerate,
  onVoice,
  isLoading,
  onFeedback,
}) => (
  <motion.div
    className="flex items-center gap-2 mt-3 opacity-0 group-hover:opacity-100"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 0 }}
    whileHover={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.2 }}
  >
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFeedback?.('up')}
            className="h-8 px-2 hover:bg-[var(--background-secondary)] text-[var(--text)]"
            aria-label="Thumbs up"
          >
            <ThumbsUp size={14} />
          </Button>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent>
        <p>Helpful</p>
      </TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFeedback?.('down')}
            className="h-8 px-2 hover:bg-[var(--background-secondary)] text-[var(--text)]"
            aria-label="Thumbs down"
          >
            <ThumbsDown size={14} />
          </Button>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent>
        <p>Not helpful</p>
      </TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={onCopy}
            className="h-8 px-2 hover:bg-[var(--background-secondary)] text-[var(--text)] relative overflow-hidden"
            aria-label="Copy message"
          >
            <Copy size={14} />
          </Button>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent>
        <p>Copy message</p>
      </TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRegenerate}
            className="h-8 px-2 hover:bg-[var(--background-secondary)] text-[var(--text)] relative overflow-hidden"
            aria-label="Regenerate response"
          >
            <motion.div
              animate={{ rotate: isLoading ? 360 : 0 }}
              transition={{ duration: 1, repeat: isLoading ? Infinity : 0, ease: 'linear' }}
            >
              <RefreshCw size={14} />
            </motion.div>
          </Button>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent>
        <p>Regenerate response</p>
      </TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={onVoice}
            className="h-8 px-2 hover:bg-[var(--background-secondary)] text-[var(--text)]"
            aria-label="Read aloud"
          >
            <Volume2 size={14} />
          </Button>
        </motion.div>
      </TooltipTrigger>
      <TooltipContent>
        <p>Read aloud</p>
      </TooltipContent>
    </Tooltip>
  </motion.div>
);
