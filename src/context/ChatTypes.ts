export interface Source {
  text?: string;
  reference?: string;
}

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: number;
  status?: 'sent' | 'delivered' | 'read' | 'error' | 'retrying';
  sources?: Source[];
  simplified?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isTyping: boolean;
  input: string;
  fontSize: number;
  showAvatars: boolean;
  isSimplifyMode: boolean;
  networkError: string | null;
  conversationStarted: boolean;
  travelInstructions: string | null;
}

export type ChatAction =
  | { type: 'SET_MESSAGES'; messages: Message[] }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'UPDATE_MESSAGE'; messageId: string; updates: Partial<Message> }
  | { type: 'SET_LOADING'; isLoading: boolean }
  | { type: 'SET_TYPING'; isTyping: boolean }
  | { type: 'SET_INPUT'; input: string }
  | { type: 'SET_FONT_SIZE'; fontSize: number }
  | { type: 'SET_SHOW_AVATARS'; showAvatars: boolean }
  | { type: 'SET_SIMPLIFY_MODE'; isSimplifyMode: boolean }
  | { type: 'SET_NETWORK_ERROR'; error: string | null }
  | { type: 'SET_CONVERSATION_STARTED'; started: boolean }
  | { type: 'SET_TRAVEL_INSTRUCTIONS'; instructions: string | null }
  | { type: 'CLEAR_CHAT' };

export interface ChatContextType extends ChatState {
  dispatch: React.Dispatch<ChatAction>;
}
