import React from 'react';
import { SkeletonChatMessage } from '@/components/ui/skeleton';

export default function RouteSkeleton() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-8 space-y-6">
        {[...Array(3)].map((_, i) => (
          <SkeletonChatMessage key={i} variant={i % 2 === 0 ? 'sent' : 'received'} />
        ))}
      </div>
    </div>
  );
}
