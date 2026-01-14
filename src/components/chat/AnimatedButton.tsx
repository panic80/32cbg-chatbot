import React from 'react';
import { Button, type ButtonProps } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type AnimatedButtonProps = ButtonProps & {
  children: React.ReactNode;
};

export const AnimatedButton: React.FC<AnimatedButtonProps> = ({
  children,
  className,
  ...props
}) => {
  return (
    <Button
      className={cn(
        'transition-colors duration-200 ease-out focus:ring-2 focus:ring-primary/20',
        className,
      )}
      {...props}
    >
      {children}
    </Button>
  );
};
