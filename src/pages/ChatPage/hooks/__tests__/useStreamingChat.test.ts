import { act, renderHook } from '@testing-library/react';
import { beforeAll, afterAll, beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useStreamingChat } from '@/pages/ChatPage/hooks/useStreamingChat';

const encoder = new TextEncoder();

type MockReader = {
  read: ReturnType<typeof vi.fn>;
};

const createMockReader = (chunks: string[]): MockReader => {
  let index = 0;
  return {
    read: vi.fn(async () => {
      if (index < chunks.length) {
        const chunk = chunks[index++];
        return { value: encoder.encode(chunk), done: false };
      }
      return { value: undefined, done: true };
    }),
  };
};

describe('useStreamingChat', () => {
  const originalFetch = global.fetch;
  const originalRAF = globalThis.requestAnimationFrame;
  const originalCancelRAF = globalThis.cancelAnimationFrame;

  beforeAll(() => {
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback): number => {
      cb(0);
      return 0;
    };
    globalThis.cancelAnimationFrame = () => {};
  });

  afterAll(() => {
    global.fetch = originalFetch;
    globalThis.requestAnimationFrame = originalRAF;
    globalThis.cancelAnimationFrame = originalCancelRAF;
  });

  let consoleErrorSpy: ReturnType<typeof vi.spyOn> | null = null;

  beforeEach(() => {
    vi.restoreAllMocks();
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    localStorage.clear();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    consoleErrorSpy?.mockRestore();
  });

  const mountHook = () => {
    const setConversationId = vi.fn();
    const setCurrentModel = vi.fn();

    const hook = renderHook(() =>
      useStreamingChat({
        conversationId: null,
        setConversationId,
        setCurrentModel,
        DEFAULT_MODEL_ID: 'gpt-5-mini',
        useRAG: true,
        shortAnswerMode: false,
        modelMode: 'fast',
      }),
    );

    return { ...hook, setConversationId, setCurrentModel };
  };

  it('streams assistant tokens through to a finalized message', async () => {
    const reader = createMockReader([
      'data: {"type":"retrieval_start"}\n',
      'data: {"type":"metadata","conversation_id":"conv-123","follow_up_questions":[{"question":"Next steps?"}]}\n',
      'data: {"type":"token","content":"Hello"}\n',
      'data: {"type":"token","content":" world"}\n',
      'data: {"type":"complete"}\n',
    ]);

    global.fetch = vi.fn(async () => ({
      ok: true,
      body: {
        getReader: () => reader,
      },
    })) as unknown as typeof fetch;

    const { result, setConversationId, setCurrentModel } = mountHook();

    await act(async () => {
      await result.current.handleStreamingChat('How are you?');
    });

    expect(setCurrentModel).toHaveBeenCalled();
    expect(setConversationId).toHaveBeenCalledWith('conv-123');

    const messages = result.current.messages;
    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({ sender: 'user', content: 'How are you?' });
    expect(messages[1]).toMatchObject({
      sender: 'assistant',
      content: 'Hello world',
    });
    expect(messages[1]?.followUpQuestions).toHaveLength(1);
    expect(result.current.pendingMessage).toBeNull();
    expect(result.current.retrievalStatus).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it('cleans up gracefully when the stream closes without a complete event', async () => {
    const reader = createMockReader([
      'data: {"type":"retrieval_start"}\n',
      'data: {"type":"token","content":"Partial"}\n',
    ]);

    global.fetch = vi.fn(async () => ({
      ok: true,
      body: {
        getReader: () => reader,
      },
    })) as unknown as typeof fetch;

    const { result } = mountHook();

    await act(async () => {
      await result.current.handleStreamingChat('Hello?');
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.pendingMessage).toBeNull();
    expect(result.current.retrievalStatus).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it('handles metadata events that omit follow up questions', async () => {
    const reader = createMockReader([
      'data: {"type":"metadata","conversation_id":"conv-999"}\n',
      'data: {"type":"token","content":"Done"}\n',
      'data: {"type":"complete"}\n',
    ]);

    global.fetch = vi.fn(async () => ({
      ok: true,
      body: {
        getReader: () => reader,
      },
    })) as unknown as typeof fetch;

    const { result, setConversationId } = mountHook();

    await act(async () => {
      await result.current.handleStreamingChat('Summarize');
    });

    expect(setConversationId).toHaveBeenCalledWith('conv-999');
    const assistant = result.current.messages[1];
    expect(assistant?.content).toBe('Done');
    expect(assistant?.followUpQuestions).toBeUndefined();
  });

  it('aborts the active fetch when the hook unmounts mid-stream', async () => {
    let capturedSignal: AbortSignal | undefined;
    global.fetch = vi.fn(async (_endpoint, init?: RequestInit) => {
      capturedSignal = init?.signal ?? undefined;
      return {
        ok: true,
        body: {
          getReader: () => ({
            read: () => new Promise(() => {}),
          }),
        },
      } as unknown as Response;
    }) as unknown as typeof fetch;

    const hook = mountHook();

    await act(async () => {
      hook.result.current.handleStreamingChat('long running');
      await Promise.resolve();
    });

    expect(capturedSignal?.aborted).toBe(false);

    hook.unmount();

    expect(capturedSignal?.aborted).toBe(true);
  });

  it('surfaces an error message when the streaming request fails', async () => {
    global.fetch = vi.fn(async () => ({
      ok: false,
      status: 500,
      statusText: 'Server error',
      text: async () => 'bad news',
    })) as unknown as typeof fetch;

    const { result } = mountHook();

    await act(async () => {
      await result.current.handleStreamingChat('trigger failure');
    });

    const messages = result.current.messages;
    expect(messages).toHaveLength(2);
    expect(messages[1]?.content).toContain('encountered an error');
    expect(result.current.isLoading).toBe(false);
    expect(result.current.retrievalStatus).toBeNull();
  });
});
