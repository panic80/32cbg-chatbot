import React, { createContext, useContext, useReducer, ReactNode } from 'react';

export type LoadingStage = 'url-scanning' | 'parsing' | 'validation' | 'complete' | 'error';

interface LoadingState {
  isLoading: boolean;
  stage: LoadingStage;
  progress: number;
  message: string;
  error?: string;
}

type LoadingAction =
  | { type: 'SET_STAGE'; payload: { stage: LoadingStage; message: string } }
  | { type: 'SET_PROGRESS'; payload: number }
  | { type: 'SET_ERROR'; payload: string }
  | { type: 'RESET' };

const initialState: LoadingState = {
  isLoading: false,
  stage: 'url-scanning',
  progress: 0,
  message: 'Initializing...',
};

const loadingReducer = (state: LoadingState, action: LoadingAction): LoadingState => {
  switch (action.type) {
    case 'SET_STAGE':
      return {
        ...state,
        isLoading: true,
        stage: action.payload.stage,
        message: action.payload.message,
        error: undefined,
      };
    case 'SET_PROGRESS':
      return {
        ...state,
        progress: Math.min(100, Math.max(0, action.payload)),
      };
    case 'SET_ERROR':
      return {
        ...state,
        stage: 'error',
        error: action.payload,
      };
    case 'RESET':
      return initialState;
    default:
      return state;
  }
};

const LoadingContext = createContext<{
  state: LoadingState;
  dispatch: React.Dispatch<LoadingAction>;
}>({
  state: initialState,
  dispatch: () => undefined,
});

export const useLoading = () => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
};

export const LoadingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(loadingReducer, initialState);

  return <LoadingContext.Provider value={{ state, dispatch }}>{children}</LoadingContext.Provider>;
};

// Utility functions for stage management
export const getStagePercentage = (stage: LoadingStage): number => {
  switch (stage) {
    case 'url-scanning':
      return 25;
    case 'parsing':
      return 50;
    case 'validation':
      return 75;
    case 'complete':
      return 100;
    case 'error':
      return 100;
    default:
      return 0;
  }
};

export const getStageMessage = (stage: LoadingStage): string => {
  switch (stage) {
    case 'url-scanning':
      return 'Analyzing URL structure...';
    case 'parsing':
      return 'Extracting URL components...';
    case 'validation':
      return 'Validating URL format...';
    case 'complete':
      return 'URL processing complete';
    case 'error':
      return 'Error processing URL';
    default:
      return 'Processing...';
  }
};
