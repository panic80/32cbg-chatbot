import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { toast } from 'sonner';
import {
  DEFAULT_CHAT_INPUT_HEIGHT_PX,
  DEFAULT_MODEL_ID,
  LOADING_DELAY_MS,
  MENU_HIGHLIGHT_DURATION_MS,
  NEW_REPLY_PILL_MARGIN_PX,
  StorageKeys,
  getModelDisplayName,
} from '@/constants';
import { MAINTENANCE_MODE } from '@/constants';
import { getLocalStorageItem, removeLocalStorageItem } from '@/utils/storage';
import { exportConversationAsMarkdown } from '@/utils/exportConversation';
import { useTheme as useThemeContext } from '@/context/ThemeContext';
import {
  useChatTheme,
  useCommandPalette,
  useDisclaimer,
  useLocalStorage,
  useMessageOperations,
  useMessageWindow,
  useModelMode,
  useScrollBehavior,
  useStreamingChat,
} from './index';

interface UseChatControllerParams {
  propTheme?: string;
  propToggleTheme?: () => void;
}

export const useChatController = ({ propTheme, propToggleTheme }: UseChatControllerParams) => {
  const [input, setInput] = useState('');
  const [currentModel, setCurrentModel] = useState(getModelDisplayName(DEFAULT_MODEL_ID));
  const [isRecording, setIsRecording] = useState(false);
  const [collapsedMessages, setCollapsedMessages] = useState<Set<string>>(new Set());
  const [showHelpDialog, setShowHelpDialog] = useState(false);
  const [useRAG] = useState(true);
  const [shortAnswerMode, setShortAnswerMode] = useLocalStorage(StorageKeys.shortAnswerMode, false);

  const [modelMode, setModelMode] = useState<'fast' | 'smart'>(() => {
    const savedModel = getLocalStorageItem(StorageKeys.selectedModel);
    return savedModel === 'gpt-5-mini' ? 'smart' : 'fast';
  });

  const [menuOpen, setMenuOpen] = useState(false);
  const [menuHighlight, setMenuHighlight] = useState<'none' | 'model' | 'short'>('none');
  const [conversationId, setConversationId] = useState<string>('');
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [inputHeight, setInputHeight] = useState<number>(DEFAULT_CHAT_INPUT_HEIGHT_PX);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const { theme: contextTheme, toggleTheme: contextToggleTheme } = useThemeContext();

  useEffect(() => {
    removeLocalStorageItem(StorageKeys.hybridSearch);
  }, []);

  useEffect(() => {
    const el = document.querySelector('[data-chat-input]') as HTMLElement | null;
    if (!el) return;

    const measure = () => {
      setInputHeight(el.getBoundingClientRect().height || DEFAULT_CHAT_INPUT_HEIGHT_PX);
    };
    measure();

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => measure());
      ro.observe(el);
    }

    window.addEventListener('resize', measure);
    return () => {
      ro?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  const pillMargin = NEW_REPLY_PILL_MARGIN_PX;

  const { messages, setMessages, pendingMessage, isLoading, retrievalStatus, handleStreamingChat } =
    useStreamingChat({
      conversationId,
      setConversationId,
      setCurrentModel,
      DEFAULT_MODEL_ID,
      useRAG,
      shortAnswerMode,
      modelMode,
      maintenanceMode: MAINTENANCE_MODE,
    });

  const {
    combinedMessages,
    visibleMessages,
    startIndex,
    canShowMore: canShowMoreMessages,
    showMore: showMoreMessages,
  } = useMessageWindow({ messages, pendingMessage });

  useEffect(() => {
    try {
      const params = new URLSearchParams(location.search);
      const q = params.get('q');
      if (q && q.trim().length > 0) {
        setInput(q.trim());
      }
    } catch {
      // no-op
    }
  }, [location.search]);

  const theme = propTheme ?? contextTheme;

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsInitialLoading(false);
    }, LOADING_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  useChatTheme(theme, propTheme);
  useModelMode(modelMode, setCurrentModel);

  const triggerMenu = useCallback((highlight: 'model' | 'short') => {
    setMenuHighlight(highlight);
    setMenuOpen(true);
  }, []);

  const handleModePillClick = useCallback(() => triggerMenu('model'), [triggerMenu]);
  const handleShortAnswerPillClick = useCallback(() => triggerMenu('short'), [triggerMenu]);

  useEffect(() => {
    if (!menuOpen && menuHighlight !== 'none') {
      setMenuHighlight('none');
    }
  }, [menuOpen, menuHighlight]);

  useEffect(() => {
    if (menuHighlight !== 'none') {
      const timer = setTimeout(() => setMenuHighlight('none'), MENU_HIGHLIGHT_DURATION_MS);
      return () => clearTimeout(timer);
    }
  }, [menuHighlight]);

  const { showDisclaimer, setShowDisclaimer } = useDisclaimer();
  const toggleTheme = propToggleTheme ?? contextToggleTheme;

  const { isAtBottom, showNewPill, scrollToBottom } = useScrollBehavior({
    scrollAreaRef,
    messages,
  });

  const handleSendMessage = useCallback(
    async (messageText?: string) => {
      const messageToSend = messageText || input.trim();
      if (!messageToSend || isLoading) return;

      if (!messageText) setInput('');

      setTimeout(scrollToBottom, 100);
      await handleStreamingChat(messageToSend);
    },
    [handleStreamingChat, input, isLoading, scrollToBottom],
  );

  const {
    commandOpen,
    setCommandOpen,
    showInlineCommand,
    setShowInlineCommand,
    selectedCommandIndex,
    handleInputChange,
    handleKeyPress,
    commands: inlineCommandOptions,
  } = useCommandPalette({
    setInput,
    onSubmit: handleSendMessage,
    setShowHelpDialog,
  });

  const handleSuggestionSelect = useCallback(
    (title: string) => {
      setInput(title);
      handleSendMessage(title);
      setInput('');
    },
    [handleSendMessage],
  );

  const toggleMessageCollapse = useCallback((messageId: string) => {
    setCollapsedMessages((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  }, []);

  const handleVoiceInput = useCallback(() => {
    setIsRecording((prev) => !prev);
  }, []);

  const handleFollowUpClick = useCallback(
    (question: string) => {
      setInput(question);
      handleSendMessage(question);
      setInput('');
    },
    [handleSendMessage],
  );

  const handleTripPlanSubmit = useCallback(
    (tripPlan: string) => {
      handleSendMessage(tripPlan);
    },
    [handleSendMessage],
  );

  const handleAcceptDisclaimer = useCallback(() => {
    setShowDisclaimer(false);
  }, [setShowDisclaimer]);

  const { copyMessage, regenerateMessage } = useMessageOperations({ setMessages });

  const exportMarkdown = useCallback(() => {
    exportConversationAsMarkdown(messages, conversationId);
    toast.success('Exported as Markdown');
  }, [conversationId, messages]);

  const clearConversation = useCallback(() => {
    setMessages([]);
    setConversationId('');
    toast.success('Conversation cleared');
  }, [setMessages]);

  const commandPaletteProps = useMemo(
    () => ({
      open: commandOpen,
      onOpenChange: setCommandOpen,
      onCommandSelect: setInput,
    }),
    [commandOpen, setCommandOpen, setInput],
  );

  const chatHeaderProps = useMemo(
    () => ({
      theme,
      toggleTheme,
      modelMode,
      setModelMode,
      onTripPlanSubmit: handleTripPlanSubmit,
      shortAnswerMode,
      setShortAnswerMode,
      onExportMarkdown: exportMarkdown,
      onClearConversation: clearConversation,
      onInsertExample: (q: string) => setInput(q),
      menuOpen,
      setMenuOpen,
      highlightModelMode: menuHighlight === 'model',
      highlightShortAnswers: menuHighlight === 'short',
    }),
    [
      clearConversation,
      exportMarkdown,
      handleTripPlanSubmit,
      menuHighlight,
      menuOpen,
      modelMode,
      setModelMode,
      setMenuOpen,
      setInput,
      setShortAnswerMode,
      shortAnswerMode,
      theme,
      toggleTheme,
    ],
  );

  const messagesPanelProps = useMemo(
    () => ({
      scrollAreaRef,
      isInitialLoading,
      messages,
      visibleMessages,
      combinedMessages,
      startIndex,
      canShowMoreMessages,
      showMoreMessages,
      collapsedMessages,
      onToggleCollapse: toggleMessageCollapse,
      onCopyMessage: copyMessage,
      onRegenerateMessage: regenerateMessage,
      onVoiceAction: handleVoiceInput,
      currentModel,
      modelMode,
      shortAnswerMode,
      isLoading,
      pendingMessage,
      onFollowUpClick: handleFollowUpClick,
      onSuggestionSelect: handleSuggestionSelect,
      retrievalStatus,
      inputHeight,
      pillMargin,
      isAtBottom,
      showNewPill,
      scrollToBottom,
      onModePillClick: handleModePillClick,
      onShortAnswerPillClick: handleShortAnswerPillClick,
    }),
    [
      canShowMoreMessages,
      collapsedMessages,
      combinedMessages,
      copyMessage,
      currentModel,
      handleFollowUpClick,
      handleModePillClick,
      handleShortAnswerPillClick,
      handleSuggestionSelect,
      handleVoiceInput,
      inputHeight,
      isAtBottom,
      isInitialLoading,
      isLoading,
      messages,
      modelMode,
      pendingMessage,
      pillMargin,
      regenerateMessage,
      retrievalStatus,
      scrollToBottom,
      shortAnswerMode,
      showMoreMessages,
      showNewPill,
      startIndex,
      toggleMessageCollapse,
      visibleMessages,
    ],
  );

  const chatInputProps = useMemo(
    () => ({
      input,
      setInput,
      handleInputChange,
      handleKeyPress,
      handleSendMessage,
      isLoading,
      showInlineCommand,
      selectedCommandIndex,
      setShowInlineCommand,
      commands: inlineCommandOptions,
      currentModel,
    }),
    [
      handleInputChange,
      handleKeyPress,
      handleSendMessage,
      inlineCommandOptions,
      input,
      isLoading,
      selectedCommandIndex,
      setInput,
      setShowInlineCommand,
      showInlineCommand,
      currentModel,
    ],
  );

  const helpDialogProps = useMemo(
    () => ({
      open: showHelpDialog,
      onOpenChange: setShowHelpDialog,
      onInsertExample: (q: string) => setInput(q),
    }),
    [setInput, setShowHelpDialog, showHelpDialog],
  );

  const disclaimerProps = useMemo(
    () => ({
      open: showDisclaimer,
      onAccept: handleAcceptDisclaimer,
    }),
    [handleAcceptDisclaimer, showDisclaimer],
  );

  return {
    theme,
    commandPaletteProps,
    chatHeaderProps,
    messagesPanelProps,
    chatInputProps,
    helpDialogProps,
    disclaimerProps,
  };
};
