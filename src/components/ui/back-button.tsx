import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { AnimatedButton } from './animated-button';
import { cn } from '@/lib/utils';

interface BackButtonProps {
  /** The route to navigate to (default: '/') */
  to?: string;
  /** Custom label for the button (default: 'Back') */
  label?: string;
  /** Additional CSS classes */
  className?: string;
  /** Size variant */
  size?: 'sm' | 'default' | 'lg';
  /** Whether to show only icon on mobile */
  iconOnlyMobile?: boolean;
  /** Custom click handler (overrides navigation) */
  onClick?: () => void;
}

/**
 * Standardized back button component for consistent navigation across the app
 */
export const BackButton: React.FC<BackButtonProps> = ({
  to = '/',
  label = 'Back',
  className,
  size = 'sm',
  iconOnlyMobile = true,
  onClick,
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      navigate(to);
    }
  };

  return (
    <AnimatedButton
      variant="ghost"
      size={size}
      onClick={handleClick}
      className={cn(
        'group flex items-center gap-2 transition-all duration-200',
        'hover:bg-muted/80 hover:shadow-sm',
        'focus-visible:ring-2 focus-visible:ring-primary',
        className,
      )}
      ripple
      aria-label={`${label} - Navigate to previous page`}
    >
      <ArrowLeft
        className={cn(
          'transition-transform duration-200 ease-out',
          'group-hover:-translate-x-1',
          size === 'sm' && 'h-4 w-4',
          size === 'default' && 'h-5 w-5',
          size === 'lg' && 'h-6 w-6',
        )}
      />
      <span className={cn('font-medium', iconOnlyMobile && 'hidden sm:inline-block')}>{label}</span>
    </AnimatedButton>
  );
};

/**
 * Page header with integrated back button
 * Useful for consistent page layouts
 */
export const PageHeaderWithBack: React.FC<{
  title: string;
  description?: string;
  backTo?: string;
  backLabel?: string;
  children?: React.ReactNode;
}> = ({ title, description, backTo, backLabel, children }) => {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-4 mb-4">
        <BackButton to={backTo} label={backLabel} />
        {children}
      </div>
      <h1 className="h1 text-fluid-4xl font-bold text-foreground mb-2">{title}</h1>
      {description && <p className="body-lg text-muted-foreground">{description}</p>}
    </div>
  );
};
