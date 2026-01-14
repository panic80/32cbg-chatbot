import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedButton } from '../components/ui/animated-button';
import { EnhancedBackButton } from '../components/ui/enhanced-back-button';
import { Skeleton, SkeletonText } from '../components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Shield, Lock, Eye, Database, Clock, AlertCircle } from 'lucide-react';

export default function PrivacyPage() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 600);
    return () => clearTimeout(timer);
  }, []);

  const privacySections = [
    {
      icon: Shield,
      title: 'Data Protection',
      content:
        'We value your privacy. This application collects minimal personal information required for policy assistance. All information provided is used solely for the purpose of delivering personalized services.',
    },
    {
      icon: Lock,
      title: 'Information Security',
      content:
        'Your data is encrypted and stored securely. We implement industry-standard security measures to protect your information from unauthorized access.',
    },
    {
      icon: Eye,
      title: 'Data Usage',
      content:
        'By using this service, you consent to our privacy practices, which may include the collection, storage, and transmission of personal information as described.',
    },
    {
      icon: Database,
      title: 'Third-Party Sharing',
      content:
        'We will not share your data with third parties without your explicit consent, except as required by law.',
    },
    {
      icon: Clock,
      title: 'Data Retention',
      content:
        'For your personal account, your interactions are saved by default for up to 18 months. You can adjust this setting to retain data for only 3 months or extend it up to 36 months.',
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-float-slow" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-float-slow delay-1000" />
        </div>

        <div className="relative z-10 max-w-5xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          {/* Back Button */}
          <div className="mb-8 animate-fade-up">
            <EnhancedBackButton to="/" label="Back" variant="minimal" />
          </div>

          {/* Header */}
          <div className="text-center mb-12 animate-fade-up">
            <div className="inline-flex items-center justify-center w-20 h-20 mb-6 bg-primary/10 rounded-full">
              <Shield className="w-10 h-10 text-primary" />
            </div>
            <h1 className="h1 text-fluid-5xl font-bold mb-6 bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
              Privacy Policy
            </h1>
            <p className="body-lg text-muted-foreground max-w-3xl mx-auto">
              Your privacy is our priority. Learn how we collect, use, and protect your personal
              information.
            </p>
          </div>

          {/* Privacy Sections */}
          <div className="grid gap-6 mb-12">
            {isLoading
              ? [...Array(5)].map((_, i) => (
                  <Card
                    key={i}
                    className="animate-fade-up"
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    <CardHeader>
                      <div className="flex items-center gap-4">
                        <Skeleton className="w-12 h-12 rounded-lg" />
                        <div className="flex-1">
                          <Skeleton className="h-6 w-48 mb-2" />
                          <SkeletonText lines={2} />
                        </div>
                      </div>
                    </CardHeader>
                  </Card>
                ))
              : privacySections.map((section, index) => {
                  const Icon = section.icon;
                  return (
                    <Card
                      key={index}
                      className="card-lift animate-fade-up hover:border-primary/20 transition-all duration-300"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <CardHeader>
                        <div className="flex items-start gap-4">
                          <div className="p-3 bg-primary/10 rounded-lg">
                            <Icon className="w-6 h-6 text-primary" />
                          </div>
                          <div className="flex-1">
                            <CardTitle className="text-xl mb-2">{section.title}</CardTitle>
                            <CardDescription className="text-base leading-relaxed">
                              {section.content}
                            </CardDescription>
                          </div>
                        </div>
                      </CardHeader>
                    </Card>
                  );
                })}
          </div>

          {/* Important Notice */}
          <Card className="mb-12 border-warning/50 bg-warning/5 animate-fade-up delay-500">
            <CardHeader>
              <div className="flex items-start gap-4">
                <AlertCircle className="w-6 h-6 text-warning mt-0.5" />
                <div>
                  <CardTitle className="text-lg mb-2">Important Notice</CardTitle>
                  <CardDescription className="text-base">
                    Even if you opt out of detailed activity tracking, temporary records may be
                    maintained for up to 72 hours to keep the service running smoothly. These
                    records are not visible in your activity log and can be manually deleted at any
                    time.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
          </Card>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-up delay-700">
            <EnhancedBackButton
              to="/"
              label="Back to Home"
              size="lg"
              variant="minimal"
              className="min-w-[150px]"
            />
            <AnimatedButton
              variant="outline"
              size="lg"
              className="min-w-[150px]"
              onClick={() => (window.location.href = '/chat')}
            >
              Contact Support
            </AnimatedButton>
          </div>

          {/* Footer */}
          <div className="mt-16 text-center animate-fade-up delay-900">
            <p className="caption text-muted-foreground">Last updated: July 7, 2025</p>
            <p className="caption text-muted-foreground mt-2">
              Please review our complete Privacy Policy for more detailed information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
