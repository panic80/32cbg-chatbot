import React from 'react';

export const BackgroundEffects = () => (
  <div className="absolute inset-0 pointer-events-none overflow-hidden">
    <div
      className="absolute w-[600px] h-[600px] rounded-full blur-3xl opacity-20"
      style={{
        background: `radial-gradient(circle at center, var(--primary) 0%, transparent 70%)`,
        left: '-10%',
        top: '-10%',
      }}
    />
    <div
      className="absolute w-[600px] h-[600px] rounded-full blur-3xl opacity-20"
      style={{
        background: `radial-gradient(circle at center, var(--primary) 0%, transparent 70%)`,
        right: '-10%',
        bottom: '-10%',
      }}
    />
  </div>
);
