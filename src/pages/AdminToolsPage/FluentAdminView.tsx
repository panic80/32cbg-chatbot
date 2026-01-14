import React, { useState, useEffect, useRef } from 'react';
import { motion, useMotionValue, useTransform } from 'framer-motion';
import { ChevronRight, ExternalLink, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

// Fluent Card Component for each admin tool section
const FluentToolCard = ({ tool, delay = 0 }) => {
  const cardRef = useRef(null);
  const [isHovered, setIsHovered] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Reveal highlight effect
  const revealX = useTransform(mouseX, (value) => value - 150);
  const revealY = useTransform(mouseY, (value) => value - 150);

  const handleMouseMove = (e) => {
    const rect = cardRef.current?.getBoundingClientRect();
    if (rect) {
      mouseX.set(e.clientX - rect.left);
      mouseY.set(e.clientY - rect.top);
    }
  };

  const Icon = tool.icon;

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        delay,
        type: 'spring',
        stiffness: 300,
        damping: 30,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseMove={handleMouseMove}
      className="relative"
    >
      {/* Acrylic Card */}
      <div
        className={cn(
          'relative overflow-hidden rounded-2xl',
          'bg-[var(--card)]/80',
          'backdrop-blur-xl backdrop-saturate-150',
          'border border-[var(--border)]/30',
          'shadow-lg hover:shadow-2xl transition-all duration-300',
        )}
      >
        {/* Reveal Highlight */}
        <motion.div
          className="absolute inset-0 opacity-0 pointer-events-none"
          style={{
            background: `radial-gradient(300px circle at ${revealX}px ${revealY}px, rgba(var(--primary-rgb), 0.1), transparent)`,
            opacity: isHovered ? 1 : 0,
            transition: 'opacity 0.2s',
          }}
        />

        {/* Card Header */}
        <div
          className="relative z-10 p-6 sm:p-8 cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {/* Icon with glow effect */}
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: delay + 0.1 }}
            className="mb-6"
          >
            <div
              className={cn(
                'w-16 h-16 rounded-2xl flex items-center justify-center',
                'bg-[var(--primary)]',
                'shadow-lg',
                isHovered && 'shadow-[var(--primary)]/30',
              )}
            >
              <Icon className="w-8 h-8 text-white" />
            </div>
          </motion.div>

          {/* Tool Info */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: delay + 0.2 }}
          >
            <h3 className="text-xl font-semibold mb-2 text-[var(--text)] flex items-center justify-between">
              {tool.title}
              <motion.div animate={{ rotate: isExpanded ? 90 : 0 }} transition={{ duration: 0.2 }}>
                <ChevronRight className="w-5 h-5" />
              </motion.div>
            </h3>
            <p className="text-[var(--text-secondary)]">{tool.description}</p>
          </motion.div>
        </div>

        {/* Expandable Links Section */}
        <motion.div
          initial={false}
          animate={{ height: isExpanded ? 'auto' : 0 }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="overflow-hidden"
        >
          <div className="px-6 pb-6 sm:px-8 sm:pb-8 border-t border-[var(--border)]/20">
            <div className="pt-4 space-y-2">
              {tool.links.map((link, idx) => (
                <motion.a
                  key={idx}
                  href={link.url}
                  target={link.external ? '_blank' : undefined}
                  rel={link.external ? 'noopener noreferrer' : undefined}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: isExpanded ? 1 : 0, x: isExpanded ? 0 : -20 }}
                  transition={{ delay: isExpanded ? idx * 0.05 : 0 }}
                  className={cn(
                    'flex items-center justify-between p-3 rounded-lg',
                    'hover:bg-[var(--background-secondary)] transition-colors',
                    'text-sm text-[var(--text)]',
                  )}
                >
                  <span>{link.name}</span>
                  {link.external ? (
                    <ExternalLink className="w-4 h-4 text-[var(--text-secondary)]" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
                  )}
                </motion.a>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Depth shadow layers */}
        <div className="absolute inset-x-0 -bottom-1 h-8 bg-black/5 dark:bg-black/10 blur-md transform scale-x-95" />
        <div className="absolute inset-x-0 -bottom-2 h-8 bg-black/3 dark:bg-black/5 blur-lg transform scale-x-90" />
      </div>
    </motion.div>
  );
};

export default function FluentAdminView({ adminTools = [] }) {
  // Ensure scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-[var(--background)] transition-colors duration-300">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-[var(--primary)] rounded-full opacity-10 blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-[var(--primary)] rounded-full opacity-10 blur-3xl animate-pulse animation-delay-2000" />
      </div>

      {/* Content */}
      <div className="relative z-10 px-4 py-8 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <Sparkles className="w-10 h-10 text-[var(--primary)]" />
              <h1 className="text-4xl font-bold text-[var(--text)]">Administrative Resources</h1>
            </div>
            <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
              Essential tools, procedures, and documentation for efficient unit administration
            </p>
          </motion.div>

          {/* Tools Grid */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid gap-6 md:grid-cols-2"
          >
            {adminTools.map((tool, index) => (
              <FluentToolCard key={tool.id} tool={tool} delay={index * 0.1} />
            ))}
          </motion.div>

          {/* Help Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-12 text-center"
          >
            <div
              className={cn(
                'inline-flex items-center gap-2 px-6 py-3 rounded-full',
                'bg-[var(--card)]/60 backdrop-blur-md',
                'border border-[var(--border)]/30',
                'text-sm text-[var(--text-secondary)]',
              )}
            >
              <span>Need help finding something?</span>
              <a
                href="mailto:g8@sent.com?subject=Admin%20Tools%20Help"
                className="text-[var(--primary)] hover:underline font-medium"
              >
                Contact Support
              </a>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Custom Styles for animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 0.1;
          }
          50% {
            opacity: 0.2;
          }
        }
        
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        
        :root {
          --primary-rgb: 59, 130, 246;
        }
      `}</style>
    </div>
  );
}
