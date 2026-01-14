import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem, setLocalStorageItem } from '@/utils/storage';
import { ChatState, ChatAction } from './ChatTypes';

export const initialState: ChatState = {
  messages: [],
  isLoading: false,
  isTyping: false,
  input: '',
  fontSize: Number(getLocalStorageItem(StorageKeys.fontSize)) || 16,
  showAvatars: true,
  isSimplifyMode: false,
  networkError: null,
  conversationStarted: false,
  travelInstructions: null,
};

export const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'SET_MESSAGES':
      return {
        ...state,
        messages: action.messages,
        conversationStarted: action.messages.length > 0,
      };

    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.message],
        conversationStarted: true,
      };

    case 'UPDATE_MESSAGE':
      return {
        ...state,
        messages: state.messages.map((msg) =>
          msg.id === action.messageId ? { ...msg, ...action.updates } : msg,
        ),
      };

    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.isLoading,
      };

    case 'SET_TYPING':
      return {
        ...state,
        isTyping: action.isTyping,
      };

    case 'SET_INPUT':
      return {
        ...state,
        input: action.input,
      };

    case 'SET_FONT_SIZE':
      setLocalStorageItem(StorageKeys.fontSize, String(action.fontSize));
      return {
        ...state,
        fontSize: action.fontSize,
      };

    case 'SET_SHOW_AVATARS':
      return {
        ...state,
        showAvatars: action.showAvatars,
      };

    case 'SET_SIMPLIFY_MODE':
      return {
        ...state,
        isSimplifyMode: action.isSimplifyMode,
      };

    case 'SET_NETWORK_ERROR':
      return {
        ...state,
        networkError: action.error,
      };

    case 'SET_CONVERSATION_STARTED':
      return {
        ...state,
        conversationStarted: action.started,
      };

    case 'SET_TRAVEL_INSTRUCTIONS':
      return {
        ...state,
        travelInstructions: action.instructions,
      };

    case 'CLEAR_CHAT':
      return {
        ...state,
        messages: [],
        conversationStarted: false,
        networkError: null,
      };

    default:
      return state;
  }
};
