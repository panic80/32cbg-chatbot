import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { AnimatedButton } from '../components/ui/animated-button';
import { EnhancedBackButton } from '../components/ui/enhanced-back-button';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Skeleton, SkeletonText } from '../components/ui/skeleton';
import { Code, Copy, CheckCircle, Server, Zap, Shield, Clock } from 'lucide-react';
import { toast } from 'sonner';

const APIPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 600);
    return () => clearTimeout(timer);
  }, []);

  const copyToClipboard = (text: string, endpoint: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(endpoint);
    toast.success('Copied to clipboard!');
    setTimeout(() => setCopiedEndpoint(null), 2000);
  };

  const endpoints = [
    {
      method: 'POST',
      path: '/api/gemini/generateContent',
      description: 'Generate AI content using Google Gemini',
      params: ['prompt', 'model', 'generationConfig'],
      example: `{
  "prompt": "Explain travel allowances",
  "model": "gemini-pro",
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 1000
  }
}`,
    },
    {
      method: 'GET',
      path: '/api/travel-instructions',
      description: 'Retrieve Canadian Forces travel instructions',
      params: [],
      example: 'No parameters required',
    },
    {
      method: 'GET',
      path: '/api/config',
      description: 'Get API configuration and available features',
      params: [],
      example: 'No parameters required',
    },
    {
      method: 'GET',
      path: '/health',
      description: 'System health check and status',
      params: [],
      example: 'No parameters required',
    },
  ];

  const features = [
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Optimized response times with intelligent caching',
    },
    {
      icon: Shield,
      title: 'Secure by Default',
      description: 'Built-in rate limiting and authentication',
    },
    {
      icon: Server,
      title: 'Always Available',
      description: '99.9% uptime with health monitoring',
    },
    {
      icon: Clock,
      title: 'Real-time Updates',
      description: 'WebSocket support for live streaming',
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/3 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-float-slow" />
          <div className="absolute bottom-0 right-1/3 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-float-slow delay-1000" />
        </div>

        <div className="relative z-10 max-w-6xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          {/* Back Button */}
          <div className="mb-8 animate-fade-up">
            <EnhancedBackButton to="/" label="Back" variant="minimal" />
          </div>

          {/* Header */}
          <div className="text-center mb-16 animate-fade-up">
            <div className="inline-flex items-center justify-center w-20 h-20 mb-6 bg-primary/10 rounded-full">
              <Code className="w-10 h-10 text-primary" />
            </div>
            <h1 className="h1 text-fluid-5xl font-bold mb-6 bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
              API Documentation
            </h1>
            <p className="body-lg text-muted-foreground max-w-2xl mx-auto">
              Simple and reliable endpoints for the Travel Instructions Chatbot with comprehensive
              documentation
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card
                  key={index}
                  className="card-lift glass animate-fade-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <CardHeader>
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                      <Icon className="w-6 h-6 text-primary" />
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardHeader>
                </Card>
              );
            })}
          </div>

          {/* API Endpoints */}
          <Card className="glass mb-12 animate-fade-up delay-300">
            <CardHeader>
              <CardTitle className="text-2xl">Available Endpoints</CardTitle>
              <CardDescription>
                All endpoints return JSON responses with appropriate HTTP status codes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="0" className="w-full">
                <TabsList className="grid w-full grid-cols-4 mb-6">
                  {endpoints.map((_, index) => (
                    <TabsTrigger key={index} value={index.toString()}>
                      Endpoint {index + 1}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {endpoints.map((endpoint, index) => (
                  <TabsContent key={index} value={index.toString()} className="space-y-6">
                    {isLoading ? (
                      <div className="space-y-4">
                        <Skeleton className="h-12 w-full" />
                        <SkeletonText lines={3} />
                        <Skeleton className="h-32 w-full" />
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {/* Endpoint Header */}
                        <div className="flex items-center justify-between flex-wrap gap-4">
                          <div className="flex items-center gap-3">
                            <Badge
                              variant={endpoint.method === 'GET' ? 'default' : 'secondary'}
                              className="font-mono"
                            >
                              {endpoint.method}
                            </Badge>
                            <code className="text-lg font-mono bg-muted px-3 py-1 rounded">
                              {endpoint.path}
                            </code>
                          </div>
                          <AnimatedButton
                            variant="outline"
                            size="sm"
                            onClick={() => copyToClipboard(endpoint.path, endpoint.path)}
                            className="gap-2"
                          >
                            {copiedEndpoint === endpoint.path ? (
                              <CheckCircle className="w-4 h-4" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                            Copy
                          </AnimatedButton>
                        </div>

                        {/* Description */}
                        <p className="text-base text-muted-foreground">{endpoint.description}</p>

                        {/* Parameters */}
                        {endpoint.params.length > 0 && (
                          <div>
                            <h4 className="font-semibold mb-3">Parameters</h4>
                            <div className="flex flex-wrap gap-2">
                              {endpoint.params.map((param, i) => (
                                <Badge key={i} variant="outline" className="font-mono">
                                  {param}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Example */}
                        <div>
                          <h4 className="font-semibold mb-3">Example</h4>
                          <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{endpoint.example}</code>
                          </pre>
                        </div>
                      </div>
                    )}
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
          </Card>

          {/* Getting Started */}
          <Card className="glass animate-fade-up delay-500">
            <CardHeader>
              <CardTitle className="text-2xl">Getting Started</CardTitle>
              <CardDescription>Quick tips to integrate with our API</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-sm font-bold text-primary">1</span>
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Authentication</h4>
                  <p className="text-muted-foreground">
                    Include your API key in the Authorization header:{' '}
                    <code className="bg-muted px-2 py-0.5 rounded text-sm">
                      Bearer YOUR_API_KEY
                    </code>
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-sm font-bold text-primary">2</span>
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Rate Limiting</h4>
                  <p className="text-muted-foreground">
                    Default rate limit is 60 requests per minute. Check response headers for current
                    usage.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-sm font-bold text-primary">3</span>
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Error Handling</h4>
                  <p className="text-muted-foreground">
                    All errors return consistent JSON with error code and message for easy
                    debugging.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Footer */}
          <div className="text-center mt-12 animate-fade-up delay-700">
            <p className="text-muted-foreground mb-4">Need help with integration?</p>
            <AnimatedButton size="lg" className="gap-2">
              <Code className="w-4 h-4" />
              View Full Documentation
            </AnimatedButton>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIPage;
