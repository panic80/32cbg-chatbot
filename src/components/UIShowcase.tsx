import React, { useState } from 'react';
import { AnimatedButton } from './ui/animated-button';
import { Button } from './ui/button';
import {
  Skeleton,
  SkeletonText,
  SkeletonButton,
  SkeletonAvatar,
  SkeletonCard,
  SkeletonChatMessage,
  SkeletonSource,
} from './ui/skeleton';
import { cn } from '@/lib/utils';

export const UIShowcase: React.FC = () => {
  const [showSkeleton, setShowSkeleton] = useState(false);

  return (
    <div className="p-8 space-y-12 max-w-6xl mx-auto">
      {/* Typography Section */}
      <section className="space-y-6">
        <h2 className="h2 text-fluid-4xl">Typography Hierarchy</h2>
        <div className="space-y-4 p-6 bg-card rounded-lg border border-border">
          <h1 className="h1">Heading 1 - Fluid Responsive</h1>
          <h2 className="h2">Heading 2 - Professional Scale</h2>
          <h3 className="h3">Heading 3 - Clear Hierarchy</h3>
          <h4 className="h4">Heading 4 - Semantic Structure</h4>
          <p className="body-lg">Large body text with optimal line height for readability</p>
          <p className="body-base">Regular body text using our custom font stack and spacing</p>
          <p className="caption text-muted-foreground">Caption text for secondary information</p>
        </div>
      </section>

      {/* Button Animations Section */}
      <section className="space-y-6">
        <h2 className="h2 text-fluid-4xl">Enhanced Button States</h2>
        <div className="flex flex-wrap gap-4 p-6 bg-card rounded-lg border border-border">
          <AnimatedButton>Ripple Effect</AnimatedButton>
          <AnimatedButton variant="secondary" glow>
            Glow Animation
          </AnimatedButton>
          <AnimatedButton variant="outline" pulse>
            Pulse Effect
          </AnimatedButton>
          <AnimatedButton variant="destructive">Error State</AnimatedButton>

          <div className="w-full mt-4" />

          <Button className="bg-success hover:bg-success-hover">Success Action</Button>
          <Button className="bg-warning hover:bg-warning-hover text-warning-foreground">
            Warning State
          </Button>
          <Button className="bg-info hover:bg-info-hover">Info Button</Button>
        </div>
      </section>

      {/* Skeleton Loaders Section */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="h2 text-fluid-4xl">Skeleton Loaders</h2>
          <Button variant="outline" size="sm" onClick={() => setShowSkeleton(!showSkeleton)}>
            {showSkeleton ? 'Show Content' : 'Show Skeletons'}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Chat Message Skeleton */}
          <div className="p-6 bg-card rounded-lg border border-border space-y-4">
            <h3 className="h5">Chat Messages</h3>
            {showSkeleton ? (
              <>
                <SkeletonChatMessage variant="sent" />
                <SkeletonChatMessage variant="received" />
              </>
            ) : (
              <div className="space-y-4">
                <div className="flex items-start gap-3 justify-end">
                  <div className="bg-primary text-primary-foreground rounded-lg p-3 max-w-xs">
                    <p className="text-sm">Hello, I need help with my travel instructions</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-muted rounded-lg p-3 max-w-xs">
                    <p className="text-sm">
                      I'd be happy to help you with your travel instructions!
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Card Skeleton */}
          <div className="p-6 bg-card rounded-lg border border-border">
            <h3 className="h5 mb-4">Content Cards</h3>
            {showSkeleton ? (
              <SkeletonCard />
            ) : (
              <div className="space-y-3">
                <div className="h-48 bg-gradient-to-br from-primary/20 to-primary/10 rounded-lg" />
                <h4 className="font-semibold">Travel Document Guidelines</h4>
                <p className="text-sm text-muted-foreground">
                  Essential information for your journey
                </p>
              </div>
            )}
          </div>

          {/* Source Citation Skeleton */}
          <div className="p-6 bg-card rounded-lg border border-border">
            <h3 className="h5 mb-4">Source Citations</h3>
            {showSkeleton ? (
              <SkeletonSource />
            ) : (
              <div className="border rounded-lg p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">📄 Travel Policy Manual</span>
                  <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                    95% match
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Members are entitled to reimbursement for travel expenses...
                </p>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Section 3.2</span>
                  <span>Page 14</span>
                </div>
              </div>
            )}
          </div>

          {/* Button Group Skeleton */}
          <div className="p-6 bg-card rounded-lg border border-border">
            <h3 className="h5 mb-4">Action Buttons</h3>
            {showSkeleton ? (
              <div className="flex gap-2">
                <SkeletonButton />
                <SkeletonButton size="sm" />
                <SkeletonAvatar size="sm" />
              </div>
            ) : (
              <div className="flex gap-2 items-center">
                <Button size="default">Continue</Button>
                <Button size="sm" variant="outline">
                  Cancel
                </Button>
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <span className="text-xs font-medium">CF</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Micro-interactions Section */}
      <section className="space-y-6">
        <h2 className="h2 text-fluid-4xl">Micro-interactions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-6 bg-card rounded-lg border border-border card-lift">
            <h3 className="h5 mb-2">Card Lift Effect</h3>
            <p className="text-sm text-muted-foreground">Hover to see elevation change</p>
          </div>

          <div className="p-6 bg-card rounded-lg border border-border hover:border-primary transition-all duration-300">
            <h3 className="h5 mb-2">Border Highlight</h3>
            <p className="text-sm text-muted-foreground">Border color on hover</p>
          </div>

          <div className="p-6 bg-gradient-to-br from-primary/10 to-primary/5 rounded-lg border border-border hover:from-primary/20 hover:to-primary/10 transition-all duration-300">
            <h3 className="h5 mb-2">Gradient Shift</h3>
            <p className="text-sm text-muted-foreground">Gradient intensity change</p>
          </div>
        </div>
      </section>

      {/* Input Animations */}
      <section className="space-y-6">
        <h2 className="h2 text-fluid-4xl">Input Enhancements</h2>
        <div className="max-w-md space-y-4 p-6 bg-card rounded-lg border border-border">
          <div>
            <input
              type="text"
              placeholder="Standard input with focus ring"
              className="w-full px-4 py-2 rounded-md border border-border bg-background input-focus-ring focus:outline-none"
            />
          </div>

          <div className="floating-label">
            <input
              type="text"
              placeholder=" "
              className="w-full px-4 py-2 rounded-md border border-border bg-background focus:outline-none focus:border-primary"
            />
            <label>Floating Label Input</label>
          </div>
        </div>
      </section>
    </div>
  );
};

export default UIShowcase;
