import * as React from 'react';
import { Button, ButtonProps } from './button';
import { cn } from '@/lib/utils';

interface AnimatedButtonProps extends ButtonProps {
  ripple?: boolean;
  glow?: boolean;
  pulse?: boolean;
  bounce?: boolean;
}

const AnimatedButton = React.forwardRef<HTMLButtonElement, AnimatedButtonProps>(
  ({ className, children, ripple = true, glow, pulse, bounce, onClick, ...props }, ref) => {
    const [ripples, setRipples] = React.useState<{ x: number; y: number; id: number }[]>([]);

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (ripple) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const id = Date.now();

        setRipples((prev) => [...prev, { x, y, id }]);

        setTimeout(() => {
          setRipples((prev) => prev.filter((r) => r.id !== id));
        }, 600);
      }

      onClick?.(e);
    };

    return (
      <Button
        ref={ref}
        className={cn(
          'relative overflow-hidden',
          glow && 'animate-glow shadow-lg shadow-primary/25',
          pulse && 'animate-pulse',
          bounce && 'hover:animate-bounce',
          className,
        )}
        onClick={handleClick}
        {...props}
      >
        {ripple &&
          ripples.map(({ x, y, id }) => (
            <span
              key={id}
              className="absolute inset-0 pointer-events-none"
              style={{
                background: `radial-gradient(circle at ${x}px ${y}px, rgba(255, 255, 255, 0.5) 0%, transparent 70%)`,
                animation: 'ripple 0.6s ease-out',
              }}
            />
          ))}
        <span className="relative z-10">{children}</span>
      </Button>
    );
  },
);

AnimatedButton.displayName = 'AnimatedButton';

export { AnimatedButton };
