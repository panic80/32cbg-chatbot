import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { exportConversationAsMarkdown, exportConversationAsJSON } from '@/utils/exportConversation';
import type { Message } from '@/types/chat';

const sampleMessages: Message[] = [
  {
    id: '1',
    content: 'Hello world',
    sender: 'user',
    timestamp: 1,
  },
  {
    id: '2',
    content: 'Response with ```code``` section',
    sender: 'assistant',
    timestamp: 2,
  },
];

describe('exportConversation utilities', () => {
  const originalCreateElement = document.createElement;

  beforeEach(() => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:url');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.createElement = originalCreateElement;
  });

  it('exports conversation as Markdown', () => {
    const clickMock = vi.fn();
    const anchorMock = {
      href: '',
      download: '',
      click: clickMock,
    } as unknown as HTMLAnchorElement;

    document.createElement = vi.fn().mockReturnValue(anchorMock);

    exportConversationAsMarkdown(sampleMessages, 'abc');

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorMock.download).toBe('conversation-abc.md');
    expect(anchorMock.href).toBe('blob:url');
    expect(clickMock).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:url');
  });

  it('exports conversation as JSON', () => {
    const clickMock = vi.fn();
    const anchorMock = {
      href: '',
      download: '',
      click: clickMock,
    } as unknown as HTMLAnchorElement;

    document.createElement = vi.fn().mockReturnValue(anchorMock);

    exportConversationAsJSON(sampleMessages, 'xyz');

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorMock.download).toBe('conversation-xyz.json');
    expect(anchorMock.href).toBe('blob:url');
    expect(clickMock).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:url');
  });
});
