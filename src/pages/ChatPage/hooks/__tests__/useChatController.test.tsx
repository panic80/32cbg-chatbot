import { renderHook, act } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '@/context/ThemeContext';
import { useChatController } from '../useChatController';
import { beforeAll, describe, expect, it, vi } from 'vitest';

class ResizeObserverMock {
  observe() {}
  disconnect() {}
}

describe('useChatController', () => {
  beforeAll(() => {
    // @ts-expect-error - assign test shim
    global.ResizeObserver = ResizeObserverMock;
  });

  it('provides structured props for chat components', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <MemoryRouter>
        <ThemeProvider>{children}</ThemeProvider>
      </MemoryRouter>
    );

    const { result } = renderHook(() => useChatController({}), { wrapper });

    expect(result.current.chatInputProps.input).toBe('');
    expect(typeof result.current.chatInputProps.handleSendMessage).toBe('function');
    expect(result.current.chatHeaderProps.theme).toBeDefined();
    expect(result.current.messagesPanelProps.isInitialLoading).toBe(true);
  });

  it('clears loading flag after simulated delay', () => {
    vi.useFakeTimers();

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <MemoryRouter>
        <ThemeProvider>{children}</ThemeProvider>
      </MemoryRouter>
    );

    const { result } = renderHook(() => useChatController({}), { wrapper });
    expect(result.current.messagesPanelProps.isInitialLoading).toBe(true);

    act(() => {
      vi.runAllTimers();
    });

    expect(result.current.messagesPanelProps.isInitialLoading).toBe(false);
    vi.useRealTimers();
  });
});
