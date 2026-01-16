import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { ChatContextType, ChatState, ChatAction, Message } from './ChatTypes';
import { chatReducer, initialState } from './ChatReducer';
import { fetchTravelInstructions } from '../api/travelInstructions';
import { generateMessageId } from '@/utils/chatUtils';

interface ExtendedChatContextType extends ChatContextType {
  generateMessageId: () => string;
}

const ChatContext = createContext<ExtendedChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Load travel instructions on mount
  useEffect(() => {
    const loadInstructions = async () => {
      try {
        const instructions = await fetchTravelInstructions();
        dispatch({ type: 'SET_TRAVEL_INSTRUCTIONS', instructions });
      } catch (error) {
        console.error('Failed to load travel instructions:', error);
        dispatch({
          type: 'SET_NETWORK_ERROR',
          error: 'Failed to load travel instructions. Please try again later.',
        });
      }
    };
    loadInstructions();
  }, []);

  // Enhanced context value with utility methods
  const contextValue: ExtendedChatContextType = {
    ...state,
    dispatch,
    generateMessageId,
  };

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
};

export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
};

export const useChat = () => {
  const context = useChatContext();

  const addMessage = useCallback(
    (message: Omit<Message, 'id' | 'timestamp'>) => {
      const fullMessage: Message = {
        ...message,
        id: context.generateMessageId(),
        timestamp: Date.now(),
        sender: message.sender,
        text: message.text,
      };
      context.dispatch({ type: 'ADD_MESSAGE', message: fullMessage });
      return fullMessage.id;
    },
    [context],
  );

  const updateMessage = useCallback(
    (messageId: string, updates: Partial<Message>) => {
      context.dispatch({ type: 'UPDATE_MESSAGE', messageId, updates });
    },
    [context],
  );

  const clearChat = useCallback(() => {
    context.dispatch({ type: 'CLEAR_CHAT' });
  }, [context]);

  const setFontSize = useCallback(
    (fontSize: number) => {
      context.dispatch({ type: 'SET_FONT_SIZE', fontSize });
    },
    [context],
  );

  const setShowAvatars = useCallback(
    (showAvatars: boolean) => {
      context.dispatch({ type: 'SET_SHOW_AVATARS', showAvatars });
    },
    [context],
  );

  const setSimplifyMode = useCallback(
    (isSimplifyMode: boolean) => {
      context.dispatch({ type: 'SET_SIMPLIFY_MODE', isSimplifyMode });
    },
    [context],
  );

  const setNetworkError = useCallback(
    (error: string | null) => {
      context.dispatch({ type: 'SET_NETWORK_ERROR', error });
    },
    [context],
  );

  return {
    state: context,
    actions: {
      addMessage,
      updateMessage,
      clearChat,
      setFontSize,
      setShowAvatars,
      setSimplifyMode,
      setNetworkError,
    },
  };
};
