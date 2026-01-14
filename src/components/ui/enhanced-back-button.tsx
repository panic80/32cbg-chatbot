import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Home, ChevronLeft } from 'lucide-react';
import { motion, useMotionValue, useSpring, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface EnhancedBackButtonProps {
  /** The route to navigate to (default: '/') */
  to?: string;
  /** Custom label for the button (default: 'Back') */
  label?: string;
  /** Additional CSS classes */
  className?: string;
  /** Size variant */
  size?: 'sm' | 'default' | 'lg';
  /** Button style variant */
  variant?: 'magnetic' | '3d' | 'glow' | 'flow' | 'minimal';
  /** Whether to show home icon instead of arrow */
  showHome?: boolean;
  /** Custom click handler (overrides navigation) */
  onClick?: () => void;
}

/**
 * Enhanced back button with modern animations and effects
 */
export const EnhancedBackButton: React.FC<EnhancedBackButtonProps> = ({
  to = '/',
  label = 'Back',
  className,
  size = 'default',
  variant = 'magnetic',
  showHome = false,
  onClick,
}) => {
  const navigate = useNavigate();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  // Motion values for magnetic effect
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const springConfig = { stiffness: 150, damping: 20, mass: 0.5 };
  const springX = useSpring(x, springConfig);
  const springY = useSpring(y, springConfig);

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      navigate(to);
    }
  };

  // Magnetic effect removed to fix flickering

  const handleMouseLeave = () => {
    setIsHovered(false);
    x.set(0);
    y.set(0);
  };

  const sizeClasses = {
    sm: 'h-9 px-3 text-sm gap-1.5',
    default: 'h-11 px-4 text-base gap-2',
    lg: 'h-14 px-6 text-lg gap-3',
  };

  const Icon = showHome ? Home : ArrowLeft;
  const iconSize = size === 'sm' ? 14 : size === 'lg' ? 20 : 16;

  // Variant-specific renders
  if (variant === 'magnetic') {
    return (
      <motion.button
        ref={buttonRef}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={handleMouseLeave}
        className={cn(
          'group relative inline-flex items-center justify-center overflow-hidden rounded-full',
          'bg-gradient-to-br from-primary/90 to-primary shadow-lg',
          'text-primary-foreground font-medium transition-colors duration-300',
          'hover:shadow-xl hover:shadow-primary/25',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
          sizeClasses[size],
          className,
        )}
        aria-label={`${label} - Navigate to previous page`}
      >
        {/* Gradient background animation */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-primary via-blue-500 to-primary opacity-0 group-hover:opacity-100"
          initial={false}
          style={{ backgroundSize: '200% 100%' }}
        />

        {/* Content */}
        <span className="relative z-10 flex items-center">
          <Icon size={iconSize} />
          <span className="hidden sm:inline-block">{label}</span>
        </span>
      </motion.button>
    );
  }

  if (variant === '3d') {
    return (
      <button
        onClick={handleClick}
        className={cn('group relative', className)}
        aria-label={`${label} - Navigate to previous page`}
      >
        <div
          className={cn(
            'relative z-10 inline-flex items-center justify-center overflow-hidden rounded-2xl',
            'bg-gradient-to-br from-primary to-primary/80 text-primary-foreground',
            'font-medium transition-colors duration-300',
            '',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
            sizeClasses[size],
          )}
        >
          <Icon size={iconSize} className="" />
          <span className="hidden sm:inline-block">{label}</span>
        </div>
        <div
          className={cn(
            'absolute inset-0 rounded-2xl bg-primary/20 transition-colors duration-300',
            '',
            'group-hover:[box-shadow:2px_2px_0_0_rgba(var(--primary-rgb),0.4),4px_4px_0_0_rgba(var(--primary-rgb),0.3),6px_6px_0_0_rgba(var(--primary-rgb),0.2)]',
          )}
        />
      </button>
    );
  }

  if (variant === 'glow') {
    return (
      <motion.button
        onClick={handleClick}
        className={cn(
          'group relative inline-flex items-center justify-center overflow-hidden rounded-full',
          'border border-primary/20 bg-background/80 backdrop-blur-sm',
          'text-primary font-medium transition-colors duration-300',
          'hover:border-primary/40 hover:bg-primary/10 hover:shadow-lg hover:shadow-primary/20',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
          sizeClasses[size],
          className,
        )}
        aria-label={`${label} - Navigate to previous page`}
      >
        {/* Animated glow */}
        <motion.div
          className="absolute -inset-1 rounded-full bg-gradient-to-r from-primary via-blue-500 to-primary opacity-0 blur-md group-hover:opacity-70"
          animate={{
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            repeatType: 'loop',
          }}
          style={{ backgroundSize: '200% 200%' }}
        />

        {/* Content */}
        <span className="relative z-10 flex items-center">
          <motion.div className="relative" transition={{ duration: 0.3 }}>
            <Icon size={iconSize} />
          </motion.div>
          <span className="hidden sm:inline-block">{label}</span>
        </span>
      </motion.button>
    );
  }

  if (variant === 'flow') {
    return (
      <button
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={cn(
          'group relative inline-flex items-center overflow-hidden rounded-full',
          'border border-border bg-background transition-colors duration-500',
          'hover:border-primary hover:shadow-lg',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
          sizeClasses[size],
          className,
        )}
        aria-label={`${label} - Navigate to previous page`}
      >
        {/* Flowing background */}
        <motion.span
          className="absolute inset-0 bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20"
          animate={{
            x: isHovered ? '100%' : '-100%',
          }}
          transition={{
            duration: 0.6,
            ease: 'easeInOut',
          }}
        />

        {/* Icon that slides in from left */}
        <AnimatePresence>
          <motion.div
            className="absolute left-3 flex items-center justify-center"
            initial={{ x: -30, opacity: 0 }}
            animate={{ x: isHovered ? 0 : -30, opacity: isHovered ? 1 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <Icon size={iconSize} className="text-primary" />
          </motion.div>
        </AnimatePresence>

        {/* Text that shifts right on hover */}
        <motion.span
          className="relative z-10 flex items-center"
          animate={{ x: isHovered ? 20 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <Icon size={iconSize} className={cn('transition-opacity', isHovered && 'opacity-0')} />
          <span className="hidden sm:inline-block">{label}</span>
        </motion.span>
      </button>
    );
  }

  // Minimal variant (default)
  return (
    <motion.button
      onClick={handleClick}
      className={cn(
        'group inline-flex items-center text-muted-foreground',
        'transition-colors duration-200 hover:text-primary',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
        size === 'sm' && 'gap-1 text-sm',
        size === 'default' && 'gap-1.5 text-base',
        size === 'lg' && 'gap-2 text-lg',
        className,
      )}
      whileHover={{ x: -4 }}
      aria-label={`${label} - Navigate to previous page`}
    >
      <motion.div
        whileHover={{ x: -2 }}
        transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      >
        <ChevronLeft size={iconSize} className="" />
      </motion.div>
      <span className="relative">
        {label}
        <motion.span
          className="absolute bottom-0 left-0 h-px w-full bg-primary"
          transition={{ duration: 0.2 }}
          style={{ transformOrigin: 'left' }}
        />
      </span>
    </motion.button>
  );
};
