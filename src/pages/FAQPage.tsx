import React, { useState, useEffect } from 'react';
import TopFAQs from '../components/TopFAQs';
import { Skeleton, SkeletonText } from '../components/ui/skeleton';
import { EnhancedBackButton } from '../components/ui/enhanced-back-button';

const FAQPage = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading state
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <div className="relative overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0">
          <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-700" />
        </div>

        <div className="relative z-10 max-w-4xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          {/* Back Button */}
          <div className="mb-8 animate-fade-up">
            <EnhancedBackButton to="/" label="Back" variant="minimal" />
          </div>

          {/* Hero Section */}
          <div className="text-center mb-12 animate-fade-up">
            <h1 className="h1 text-fluid-5xl font-bold mb-6 bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
              Frequently Asked Questions
            </h1>
            <p className="body-lg text-muted-foreground max-w-2xl mx-auto">
              Find answers to common questions about Canadian Forces travel instructions and
              policies
            </p>
          </div>

          {/* FAQ Content */}
          <div className="glass rounded-2xl p-8 backdrop-blur-xl animate-fade-up delay-200">
            {isLoading ? (
              <div className="space-y-6">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="border-b border-border/50 pb-6 last:border-0">
                    <Skeleton className="h-6 w-3/4 mb-3" />
                    <SkeletonText lines={2} />
                  </div>
                ))}
              </div>
            ) : (
              <TopFAQs visible={true} />
            )}
          </div>

          {/* Help Section */}
          <div className="mt-12 text-center animate-fade-up delay-300">
            <p className="body-base text-muted-foreground mb-4">
              Can't find what you're looking for?
            </p>
            <a
              href="/chat"
              className="inline-flex items-center gap-2 px-6 py-3 bg-primary hover:bg-primary-hover text-primary-foreground rounded-lg font-medium transition-all hover:scale-105 hover:shadow-lg"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              Ask our Chat Assistant
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FAQPage;
