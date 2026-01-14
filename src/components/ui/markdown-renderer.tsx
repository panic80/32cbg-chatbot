import React from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { cn } from '@/lib/utils';
import { CopyButton } from '@/components/ui/copy-button';

interface MarkdownRendererProps {
  children: string;
}

export const MarkdownRenderer = React.memo(function MarkdownRenderer({
  children,
}: MarkdownRendererProps) {
  return (
    <div className="space-y-3">
      <Markdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children}
      </Markdown>
    </div>
  );
});

interface CodeBlockProps extends React.HTMLAttributes<HTMLPreElement> {
  children: React.ReactNode;
  className?: string;
  language: string;
}

const CodeBlock = ({ children, className, language, ...restProps }: CodeBlockProps) => {
  const code = typeof children === 'string' ? children : childrenTakeAllStringContents(children);

  const preClass = cn(
    'overflow-x-scroll rounded-md border bg-background/50 p-4 font-mono text-sm [scrollbar-width:none]',
    className,
  );

  return (
    <div className="group/code relative mb-4">
      <pre className={preClass} {...restProps}>
        <code>{code}</code>
      </pre>

      <div className="invisible absolute right-2 top-2 flex space-x-1 rounded-lg p-1 opacity-0 transition-all duration-200 group-hover/code:visible group-hover/code:opacity-100">
        <CopyButton content={code} copyMessage="Copied code to clipboard" />
      </div>
    </div>
  );
};

function childrenTakeAllStringContents(element: React.ReactNode): string {
  if (typeof element === 'string') {
    return element;
  }

  if (React.isValidElement(element) && element.props?.children) {
    const children = element.props.children;

    if (Array.isArray(children)) {
      return children.map((child) => childrenTakeAllStringContents(child)).join('');
    } else {
      return childrenTakeAllStringContents(children);
    }
  }

  return '';
}

const COMPONENTS = {
  h1: withClass('h1', 'text-xl sm:text-2xl font-semibold'),
  h2: withClass('h2', 'font-semibold text-lg sm:text-xl'),
  h3: withClass('h3', 'font-semibold text-base sm:text-lg'),
  h4: withClass('h4', 'font-semibold text-base'),
  h5: withClass('h5', 'font-medium'),
  strong: withClass('strong', 'font-semibold'),
  a: withClass('a', 'text-primary underline underline-offset-2'),
  blockquote: withClass('blockquote', 'border-l-2 border-primary pl-4'),
  code: ({
    children,
    className,
    _node,
    ...rest
  }: {
    children: React.ReactNode;
    className?: string;
    _node?: unknown;
    [key: string]: unknown;
  }) => {
    const match = /language-(\w+)/.exec(className || '');
    return match ? (
      <CodeBlock language={match[1]} {...(rest as unknown as CodeBlockProps)}>
        {children}
      </CodeBlock>
    ) : (
      <code
        className={cn(
          'font-mono [:not(pre)>&]:rounded-md [:not(pre)>&]:bg-background/50 [:not(pre)>&]:px-1 [:not(pre)>&]:py-0.5',
        )}
        {...(rest as Record<string, unknown>)}
      >
        {children}
      </code>
    );
  },
  pre: ({ children }: { children: React.ReactNode }) => children,
  ol: withClass('ol', 'list-decimal space-y-2 pl-6'),
  ul: withClass('ul', 'list-disc space-y-2 pl-6'),
  li: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <li className="my-1.5" {...props}>
      {children}
    </li>
  ),
  table: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div className="relative w-full my-4">
      <div className="overflow-x-auto touch-pan-x rounded-md border border-foreground/20 [-webkit-overflow-scrolling:touch]">
        <table className="min-w-[600px] w-full border-collapse" {...props}>
          {children}
        </table>
      </div>
    </div>
  ),
  th: withClass(
    'th',
    'border border-foreground/20 px-3 py-2 sm:px-4 text-left font-bold whitespace-nowrap [&[align=center]]:text-center [&[align=right]]:text-right',
  ),
  td: withClass(
    'td',
    'border border-foreground/20 px-3 py-2 sm:px-4 text-left [&[align=center]]:text-center [&[align=right]]:text-right',
  ),
  tr: withClass('tr', 'm-0 border-t p-0 even:bg-muted'),
  p: withClass('p', 'whitespace-pre-wrap text-base sm:text-base'),
  hr: withClass('hr', 'border-foreground/20'),
};

function withClass(Tag: keyof JSX.IntrinsicElements, classes: string) {
  const Component = ({ _node, ...props }: { _node?: unknown; [key: string]: unknown }) => (
    <Tag className={classes} {...(props as Record<string, unknown>)} />
  );
  Component.displayName = Tag;
  return Component;
}

export default MarkdownRenderer;
