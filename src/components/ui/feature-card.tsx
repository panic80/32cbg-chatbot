import React from 'react';
import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface FeatureCardProps {
  icon: LucideIcon;
  title: React.ReactNode;
  description?: React.ReactNode;
  badge?: React.ReactNode;
  footer?: React.ReactNode;
  align?: 'center' | 'start';
  variant?: 'elevated' | 'minimal';
  disabled?: boolean;
  disabledLabel?: React.ReactNode;
  className?: string;
  titleClassName?: string;
  descriptionClassName?: string;
  footerClassName?: string;
  iconWrapperClassName?: string;
  iconClassName?: string;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({
  icon: Icon,
  title,
  description,
  badge,
  footer,
  align = 'center',
  variant = 'elevated',
  disabled = false,
  disabledLabel = 'Unavailable',
  className,
  titleClassName,
  descriptionClassName,
  footerClassName,
  iconWrapperClassName,
  iconClassName,
}) => {
  const isCentered = align === 'center';

  if (variant === 'minimal') {
    const content = (
      <div
        className={cn(
          'flex flex-col gap-2',
          isCentered ? 'items-center text-center' : 'items-start text-left',
          className,
        )}
      >
        {badge && (
          <span className={cn('lpt-minimal-card-badge', titleClassName && 'mb-1')}>{badge}</span>
        )}
        <Icon className={cn('lpt-minimal-card-icon', iconClassName)} aria-hidden="true" />
        <span className={cn('lpt-minimal-card-label', titleClassName)}>{title}</span>
        {description && (
          <span className={cn('lpt-minimal-card-subtitle', descriptionClassName)}>
            {description}
          </span>
        )}
        {footer && (
          <span className={cn('text-xs text-[var(--text-secondary)]', footerClassName)}>
            {footer}
          </span>
        )}
      </div>
    );

    if (!disabled) {
      return content;
    }

    return (
      <div className="relative">
        <div className="absolute inset-0 bg-[var(--background)]/50 rounded-2xl z-10 flex items-center justify-center">
          <div className="bg-[var(--primary)] text-white px-4 py-2 rounded-full font-medium text-sm shadow-lg animate-pulse">
            {disabledLabel}
          </div>
        </div>
        {content}
      </div>
    );
  }

  const elevatedContent = (
    <div
      className={cn(
        'flex flex-col gap-6 sm:gap-8',
        isCentered ? 'items-center text-center' : 'items-start text-left',
        className,
      )}
    >
      <div className="relative">
        <div className="absolute inset-0 bg-[var(--primary)] opacity-20 rounded-full blur-xl transform group-hover:scale-150 transition-transform duration-300" />
        <div
          className={cn(
            'relative flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-[var(--background)] shadow-inner transform transition-all duration-300 group-hover:scale-110',
            iconWrapperClassName,
          )}
        >
          <Icon
            className={cn('w-8 h-8 sm:w-10 sm:h-10 text-[var(--primary)]', iconClassName)}
            aria-hidden="true"
          />
        </div>
      </div>

      <div className={cn('space-y-3 sm:space-y-4', isCentered ? 'items-center' : 'items-start')}>
        <div
          className={cn(
            'flex items-center gap-2 text-[var(--text)]',
            isCentered ? 'justify-center' : 'justify-start',
          )}
        >
          <h3 className={cn('text-lg sm:text-xl md:text-2xl font-semibold', titleClassName)}>
            {title}
          </h3>
          {badge}
        </div>
        {description && (
          <div
            className={cn(
              'text-sm sm:text-base text-[var(--text)]/80 leading-relaxed',
              descriptionClassName,
            )}
          >
            {description}
          </div>
        )}
        {footer && (
          <div
            className={cn(
              'text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed',
              footerClassName,
            )}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  );

  if (!disabled) {
    return elevatedContent;
  }

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-[var(--background)]/60 rounded-2xl z-10 flex items-center justify-center">
        <div className="bg-[var(--primary)] text-white px-4 py-2 rounded-full font-medium text-sm shadow-lg animate-pulse">
          {disabledLabel}
        </div>
      </div>
      {elevatedContent}
    </div>
  );
};
