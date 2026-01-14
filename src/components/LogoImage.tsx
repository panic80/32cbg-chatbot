import React from 'react';
import logoImg from '../assets/logo.png';

const LogoImageBase = ({ className = '', size = 'md', fitParent = false }) => {
  const sizes = {
    xs: { width: 40, height: 48 },
    sm: { width: 60, height: 72 },
    md: { width: 100, height: 120 },
    lg: { width: 150, height: 180 },
    xl: { width: 200, height: 240 },
  };

  const { width, height } = sizes[size] || sizes.md;

  // If fitParent is true, let the image scale to its container's height
  if (fitParent) {
    return (
      <img
        src={logoImg}
        alt="32 CBG Badge"
        className={className}
        style={{
          objectFit: 'contain',
          height: '100%',
          width: 'auto',
          maxHeight: '100%',
          display: 'block',
        }}
      />
    );
  }

  return (
    <img
      src={logoImg}
      alt="32 CBG Badge"
      width={width}
      height={height}
      className={className}
      style={{ objectFit: 'contain', display: 'block' }}
    />
  );
};

// Prevent unnecessary re-renders when props haven't changed
const LogoImage = React.memo(LogoImageBase);

export default LogoImage;
