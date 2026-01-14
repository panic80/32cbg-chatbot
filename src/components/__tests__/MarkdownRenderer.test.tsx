import React from 'react';
import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MarkdownRenderer } from '@/components/ui/markdown-renderer';

describe('MarkdownRenderer', () => {
  it('renders plain text when encountering acronyms', () => {
    const { container } = render(
      <MarkdownRenderer>{'CAF members rely on official CAF guidance.'}</MarkdownRenderer>,
    );

    const paragraph = container.querySelector('p');
    expect(paragraph).not.toBeNull();
    expect(paragraph?.textContent).toContain('CAF members rely on official CAF guidance.');
    expect(container.querySelector('.cursor-help')).toBeNull();
  });

  it('renders lists without glossary tooltip wrappers', () => {
    const markdown = ['* CAF', '* TD', '* POMV'].join('\n');
    const { container } = render(<MarkdownRenderer>{markdown}</MarkdownRenderer>);

    const list = container.querySelector('ul');
    expect(list).not.toBeNull();
    expect(list?.querySelectorAll('li')).toHaveLength(3);
    expect(container.querySelector('.cursor-help')).toBeNull();
  });
});
