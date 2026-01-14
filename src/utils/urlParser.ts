import { LoadingStage } from '../context/LoadingContext';

export interface ParsedUrlResult {
  protocol: string;
  hostname: string;
  pathname: string;
  searchParams: Record<string, string>;
  hash: string;
}

export interface UrlParserEvents {
  onStageChange: (stage: LoadingStage, message: string) => void;
  onError: (error: string) => void;
  onComplete: (result: ParsedUrlResult) => void;
}

export class UrlParser {
  private url: string;
  private events: UrlParserEvents;

  constructor(url: string, events: UrlParserEvents) {
    this.url = url;
    this.events = events;
  }

  async parse(): Promise<void> {
    try {
      // Stage 1: URL Scanning
      this.events.onStageChange('url-scanning', 'Analyzing URL structure...');
      await this.simulateWork(500); // Simulate network delay

      if (!this.url) {
        throw new Error('URL cannot be empty');
      }

      // Stage 2: Parsing
      this.events.onStageChange('parsing', 'Extracting URL components...');
      await this.simulateWork(500);

      const parsedUrl = new URL(this.url);

      // Stage 3: Validation
      this.events.onStageChange('validation', 'Validating URL format...');
      await this.simulateWork(500);

      this.validateUrl(parsedUrl);

      // Stage 4: Complete
      const result: ParsedUrlResult = {
        protocol: parsedUrl.protocol.replace(':', ''),
        hostname: parsedUrl.hostname,
        pathname: parsedUrl.pathname,
        searchParams: Object.fromEntries(parsedUrl.searchParams.entries()),
        hash: parsedUrl.hash.replace('#', ''),
      };

      this.events.onStageChange('complete', 'URL processing complete');
      this.events.onComplete(result);
    } catch (error) {
      this.events.onError(error instanceof Error ? error.message : 'Unknown error occurred');
    }
  }

  private validateUrl(parsedUrl: URL): void {
    if (!parsedUrl.protocol || !parsedUrl.hostname) {
      throw new Error('Invalid URL: missing protocol or hostname');
    }

    // Validate protocol
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      throw new Error('Invalid URL: protocol must be http or https');
    }

    // Validate hostname format
    const hostnameRegex = /^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$/;
    if (!hostnameRegex.test(parsedUrl.hostname)) {
      throw new Error('Invalid URL: hostname format is invalid');
    }
  }

  private async simulateWork(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Utility method to use the parser
  static async parseUrl(
    url: string,
    onStageChange: (stage: LoadingStage, message: string) => void,
    onError: (error: string) => void,
    onComplete: (result: ParsedUrlResult) => void,
  ): Promise<void> {
    const parser = new UrlParser(url, {
      onStageChange,
      onError,
      onComplete,
    });
    await parser.parse();
  }
}
