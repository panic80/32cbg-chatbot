declare module '@/components/LogoImage' {
  import { ComponentType } from 'react';

  export interface LogoImageProps {
    className?: string;
    size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
    fitParent?: boolean;
  }

  const LogoImage: ComponentType<LogoImageProps>;
  export default LogoImage;
}
