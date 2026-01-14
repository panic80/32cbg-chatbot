import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { AnimatedButton } from '../components/ui/animated-button';
import { EnhancedBackButton } from '../components/ui/enhanced-back-button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Progress } from '../components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Skeleton, SkeletonText, SkeletonCard } from '../components/ui/skeleton';
import {
  FileText,
  Shield,
  Clock,
  CheckCircle,
  AlertCircle,
  Info,
  Download,
  Upload,
  RefreshCw,
  Lock,
} from 'lucide-react';

const SCIPPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [processingStatus, setProcessingStatus] = useState('idle');

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const scidCategories = [
    {
      level: 'SCID Level 1',
      description: 'Public information - No restrictions',
      color: 'bg-green-500',
      icon: CheckCircle,
      examples: ['Public announcements', 'General policies', 'Public-facing documents'],
    },
    {
      level: 'SCID Level 2',
      description: 'Internal use only - Limited distribution',
      color: 'bg-yellow-500',
      icon: AlertCircle,
      examples: ['Internal memos', 'Draft policies', 'Working documents'],
    },
    {
      level: 'SCID Level 3',
      description: 'Confidential - Restricted access',
      color: 'bg-orange-500',
      icon: Shield,
      examples: ['Personnel records', 'Financial data', 'Strategic plans'],
    },
    {
      level: 'SCID Level 4',
      description: 'Secret - Highly restricted',
      color: 'bg-red-500',
      icon: Lock,
      examples: ['Classified operations', 'Sensitive intelligence', 'Critical infrastructure'],
    },
  ];

  const features = [
    {
      title: 'Automated Classification',
      description: 'AI-powered document classification based on content analysis',
      icon: FileText,
      status: 'active',
    },
    {
      title: 'Real-time Monitoring',
      description: 'Continuous monitoring of document access and distribution',
      icon: Clock,
      status: 'active',
    },
    {
      title: 'Compliance Tracking',
      description: 'Ensure adherence to SCIP policies and regulations',
      icon: Shield,
      status: 'coming-soon',
    },
    {
      title: 'Audit Trail',
      description: 'Complete audit history for all classified documents',
      icon: CheckCircle,
      status: 'active',
    },
  ];

  const processingTimeoutsRef = useRef<number[]>([]);

  // Cleanup processing timeouts on unmount
  useEffect(() => {
    return () => {
      processingTimeoutsRef.current.forEach((id) => clearTimeout(id));
    };
  }, []);

  const simulateProcessing = () => {
    // Clear any existing timeouts
    processingTimeoutsRef.current.forEach((id) => clearTimeout(id));
    processingTimeoutsRef.current = [];

    setProcessingStatus('processing');
    const timeout1 = window.setTimeout(() => {
      setProcessingStatus('complete');
      const timeout2 = window.setTimeout(() => setProcessingStatus('idle'), 3000);
      processingTimeoutsRef.current.push(timeout2);
    }, 2000);
    processingTimeoutsRef.current.push(timeout1);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-red-500/5 rounded-full blur-3xl animate-float-slow" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-orange-500/5 rounded-full blur-3xl animate-float-slow delay-1000" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          {/* Back Button */}
          <div className="mb-8 animate-fade-up">
            <EnhancedBackButton to="/" label="Back" variant="minimal" />
          </div>

          {/* Header */}
          <div className="text-center mb-12 animate-fade-up">
            <div className="inline-flex items-center justify-center w-20 h-20 mb-6 bg-red-500/10 rounded-full">
              <Shield className="w-10 h-10 text-red-600" />
            </div>
            <h1 className="h1 text-fluid-5xl font-bold mb-6 bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
              SCIP Interface
            </h1>
            <p className="body-lg text-muted-foreground max-w-3xl mx-auto">
              Security Classification and Information Protection system for managing classified
              documents and ensuring compliance with security protocols
            </p>
          </div>

          {/* Alert Banner */}
          <Alert className="mb-8 border-orange-500/50 bg-orange-500/5 animate-fade-up delay-200">
            <AlertCircle className="h-4 w-4 text-orange-600" />
            <AlertTitle>Security Notice</AlertTitle>
            <AlertDescription>
              This interface handles classified information. All actions are logged and monitored
              for security compliance.
            </AlertDescription>
          </Alert>

          {/* Main Content Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="classification">Classification</TabsTrigger>
              <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
              <TabsTrigger value="reports">Reports</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-8 animate-fade-up">
              {/* SCID Levels */}
              <div>
                <h2 className="h3 text-2xl mb-6">Security Classification Levels</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {isLoading
                    ? [...Array(4)].map((_, i) => <SkeletonCard key={i} />)
                    : scidCategories.map((category, index) => {
                        const Icon = category.icon;
                        return (
                          <Card
                            key={index}
                            className="card-lift glass border-border/50 animate-fade-up"
                            style={{ animationDelay: `${index * 100}ms` }}
                          >
                            <CardHeader>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <div className={`w-3 h-3 rounded-full ${category.color}`} />
                                  <CardTitle className="text-lg">{category.level}</CardTitle>
                                </div>
                                <Icon className="w-5 h-5 text-muted-foreground" />
                              </div>
                              <CardDescription>{category.description}</CardDescription>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-2">
                                <p className="text-sm font-medium">Examples:</p>
                                <ul className="text-sm text-muted-foreground space-y-1">
                                  {category.examples.map((example, i) => (
                                    <li key={i} className="flex items-center gap-2">
                                      <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full" />
                                      {example}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                </div>
              </div>

              {/* Features Grid */}
              <div>
                <h2 className="h3 text-2xl mb-6">System Features</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {features.map((feature, index) => {
                    const Icon = feature.icon;
                    return (
                      <Card
                        key={index}
                        className="glass animate-fade-up"
                        style={{ animationDelay: `${(index + 4) * 100}ms` }}
                      >
                        <CardHeader>
                          <div className="flex items-center justify-between mb-2">
                            <Icon className="w-8 h-8 text-primary" />
                            {feature.status === 'coming-soon' && (
                              <Badge variant="secondary" className="text-xs">
                                Coming Soon
                              </Badge>
                            )}
                          </div>
                          <CardTitle className="text-base">{feature.title}</CardTitle>
                          <CardDescription className="text-sm">
                            {feature.description}
                          </CardDescription>
                        </CardHeader>
                      </Card>
                    );
                  })}
                </div>
              </div>
            </TabsContent>

            {/* Classification Tab */}
            <TabsContent value="classification" className="space-y-6 animate-fade-up">
              <Card className="glass">
                <CardHeader>
                  <CardTitle>Document Classification Tool</CardTitle>
                  <CardDescription>
                    Upload documents for automatic security classification analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="border-2 border-dashed border-border rounded-lg p-12 text-center">
                    <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-lg font-medium mb-2">Drop files here or click to upload</p>
                    <p className="text-sm text-muted-foreground mb-4">
                      Supported formats: PDF, DOC, DOCX, TXT
                    </p>
                    <AnimatedButton onClick={simulateProcessing}>
                      <Upload className="w-4 h-4 mr-2" />
                      Select Files
                    </AnimatedButton>
                  </div>

                  {processingStatus !== 'idle' && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Processing document...</span>
                        <span className="text-sm text-muted-foreground">
                          {processingStatus === 'processing' ? 'Analyzing...' : 'Complete'}
                        </span>
                      </div>
                      <Progress
                        value={processingStatus === 'processing' ? 60 : 100}
                        className="h-2"
                      />
                      {processingStatus === 'complete' && (
                        <Alert className="border-green-500/50 bg-green-500/5">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <AlertTitle>Classification Complete</AlertTitle>
                          <AlertDescription>
                            Document classified as SCID Level 2 - Internal use only
                          </AlertDescription>
                        </Alert>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Monitoring Tab */}
            <TabsContent value="monitoring" className="space-y-6 animate-fade-up">
              <Card className="glass">
                <CardHeader>
                  <CardTitle>Real-time Activity Monitor</CardTitle>
                  <CardDescription>
                    Track document access and security events in real-time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isLoading ? (
                    <div className="space-y-4">
                      {[...Array(5)].map((_, i) => (
                        <div key={i} className="flex items-center gap-4">
                          <Skeleton className="w-12 h-12 rounded-full" />
                          <div className="flex-1">
                            <Skeleton className="h-4 w-48 mb-2" />
                            <Skeleton className="h-3 w-32" />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="text-center py-12">
                        <RefreshCw className="w-12 h-12 mx-auto mb-4 text-muted-foreground animate-spin" />
                        <p className="text-muted-foreground">Monitoring active...</p>
                        <p className="text-sm text-muted-foreground mt-2">
                          No security events detected
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Reports Tab */}
            <TabsContent value="reports" className="space-y-6 animate-fade-up">
              <Card className="glass">
                <CardHeader>
                  <CardTitle>Security Reports</CardTitle>
                  <CardDescription>Generate and download compliance reports</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {[
                      'Monthly Security Audit',
                      'Access Control Report',
                      'Classification Summary',
                      'Compliance Report',
                    ].map((report, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <FileText className="w-5 h-5 text-muted-foreground" />
                          <span className="font-medium">{report}</span>
                        </div>
                        <AnimatedButton variant="ghost" size="sm">
                          <Download className="w-4 h-4" />
                        </AnimatedButton>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Footer Help Section */}
          <div className="mt-12 text-center animate-fade-up delay-700">
            <Card className="glass inline-block">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Info className="w-5 h-5 text-blue-600" />
                  <CardTitle className="text-lg">Need Help?</CardTitle>
                </div>
                <CardDescription>
                  Contact the Security Office for assistance with SCIP classifications
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SCIPPage;
