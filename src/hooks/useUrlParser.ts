import { useCallback } from 'react';
import { useLoading } from '../context/LoadingContext';
import { UrlParser, ParsedUrlResult } from '../utils/urlParser';

interface UseUrlParserResult {
  parseUrl: (url: string) => Promise<ParsedUrlResult | undefined>;
  isLoading: boolean;
  error: string | undefined;
}

export const useUrlParser = (): UseUrlParserResult => {
  const { state, dispatch } = useLoading();

  const parseUrl = useCallback(
    async (url: string): Promise<ParsedUrlResult | undefined> => {
      dispatch({ type: 'RESET' });

      return new Promise((resolve) => {
        UrlParser.parseUrl(
          url,
          (stage, message) => {
            dispatch({
              type: 'SET_STAGE',
              payload: { stage, message },
            });
          },
          (error) => {
            dispatch({ type: 'SET_ERROR', payload: error });
            resolve(undefined);
          },
          (result) => {
            dispatch({
              type: 'SET_STAGE',
              payload: { stage: 'complete', message: 'URL processing complete' },
            });
            resolve(result);
          },
        );
      });
    },
    [dispatch],
  );

  return {
    parseUrl,
    isLoading: state.isLoading,
    error: state.error,
  };
};

// Usage example:
/*
const MyComponent = () => {
  const { parseUrl, isLoading, error } = useUrlParser();

  const handleUrlSubmit = async (url: string) => {
    const result = await parseUrl(url);
    if (result) {
      // Handle successful parsing
    }
  };

  return (
    <>
      {isLoading && <LoadingScreen />}
      // Rest of your component
    </>
  );
};
*/
