import { motion } from 'framer-motion';

export const TypingIndicator = () => (
  <motion.div
    className="flex items-center gap-1 px-1"
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.8 }}
  >
    {[0, 1, 2].map((i) => (
      <motion.div
        key={i}
        className="w-2 h-2 bg-[var(--primary)] rounded-full"
        animate={{
          y: [0, -8, 0],
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          duration: 1.2,
          repeat: Infinity,
          delay: i * 0.15,
          ease: 'easeInOut',
        }}
      />
    ))}
  </motion.div>
);
