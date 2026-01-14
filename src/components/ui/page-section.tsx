import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface PageSectionProps extends Omit<React.HTMLAttributes<HTMLElement>, 'title'> {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  center?: boolean;
  description?: React.ReactNode;
  as?: React.ElementType;
}

export const PageSection = forwardRef<HTMLElement, PageSectionProps>(
  (
    {
      title,
      subtitle,
      description,
      center = false,
      className,
      as: Component = 'section',
      children,
      ...rest
    },
    ref,
  ) => {
    const alignment = center ? 'text-center items-center' : 'text-left items-start';
    const ElementComponent = Component as React.ElementType;

    return (
      <ElementComponent ref={ref} className={cn('py-16 sm:py-20', className)} {...rest}>
        <div className={cn('flex flex-col gap-4 sm:gap-6 mb-10 sm:mb-12', alignment)}>
          {subtitle && (
            <span className={cn('text-sm font-medium tracking-wide text-[var(--primary)]')}>
              {subtitle}
            </span>
          )}
          {title && (
            <h2
              className={cn(
                'text-2xl sm:text-3xl md:text-4xl font-bold text-[var(--text)]',
                center && 'mx-auto max-w-3xl',
              )}
            >
              {title}
            </h2>
          )}
          {description && (
            <p
              className={cn(
                'text-sm sm:text-base text-[var(--text-secondary)] leading-relaxed',
                center && 'mx-auto max-w-2xl',
              )}
            >
              {description}
            </p>
          )}
        </div>
        {children}
      </ElementComponent>
    );
  },
);

PageSection.displayName = 'PageSection';
