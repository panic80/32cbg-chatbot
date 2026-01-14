import { apiClient, ApiError } from '@/api/client';
import { formatText } from '../utils/chatUtils';
import { ChatError, ChatErrorType } from '../utils/chatErrors';

export const CACHE_CONFIG = {
  DB_NAME: 'travel-instructions-cache',
  STORE_NAME: 'instructions',
  CACHE_KEY: 'travel-data',
  CACHE_DURATION: 24 * 60 * 60 * 1000, // 24 hours
} as const;

export const DEFAULT_INSTRUCTIONS = `
Canadian Forces Temporary Duty Travel Instructions

1. General Information
1.1 These instructions apply to all Canadian Forces members on temporary duty travel.
1.2 Travel arrangements should be made in the most economical manner possible.

2. Authorization
2.1 All temporary duty travel must be authorized in advance.
2.2 Travel claims must be submitted within 30 days of completion of travel.

3. Transportation
3.1 The most economical means of transportation should be used.
3.2 Use of private motor vehicle requires prior approval.

4. Accommodation
4.1 Government approved accommodations should be used when available.
4.2 Commercial accommodations require receipts for reimbursement.

5. Meals and Incidentals
5.1 Meal allowances are provided for duty travel.
5.2 Incidental expenses are covered as per current rates.
`;

export const initDB = (): Promise<IDBDatabase> =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(CACHE_CONFIG.DB_NAME, 1);

    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB'));
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const target = event.target as IDBOpenDBRequest;
      const db = target.result;
      if (!db.objectStoreNames.contains(CACHE_CONFIG.STORE_NAME)) {
        db.createObjectStore(CACHE_CONFIG.STORE_NAME);
      }
    };
  });

export const getCachedData = async (): Promise<string | null> => {
  try {
    const db = await initDB();
    return await new Promise<string | null>((resolve, reject) => {
      const transaction = db.transaction(CACHE_CONFIG.STORE_NAME, 'readonly');
      const store = transaction.objectStore(CACHE_CONFIG.STORE_NAME);
      const request = store.get(CACHE_CONFIG.CACHE_KEY);

      request.onerror = () => reject(request.error ?? new Error('Failed to read from IndexedDB'));
      request.onsuccess = () => {
        const data = request.result as { content: string; timestamp: number } | undefined;
        if (data && Date.now() - data.timestamp < CACHE_CONFIG.CACHE_DURATION) {
          resolve(data.content);
        } else {
          resolve(null);
        }
      };
    });
  } catch (error) {
    console.error('Error accessing cache:', error);
    return null;
  }
};

export const setCachedData = async (content: string): Promise<void> => {
  try {
    const db = await initDB();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(CACHE_CONFIG.STORE_NAME, 'readwrite');
      const store = transaction.objectStore(CACHE_CONFIG.STORE_NAME);
      const request = store.put(
        {
          content,
          timestamp: Date.now(),
        },
        CACHE_CONFIG.CACHE_KEY,
      );

      request.onerror = () => reject(request.error ?? new Error('Failed to write to IndexedDB'));
      request.onsuccess = () => resolve();
    });
  } catch (error) {
    console.error('Error updating cache:', error);
  }
};

export const fetchWithRetry = async (apiUrl: string, maxRetries = 3): Promise<Response> => {
  let retries = maxRetries;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const response = await apiClient.request(apiUrl, {
        headers: {
          Accept: 'application/json',
          'Cache-Control': 'no-cache',
        },
        parseErrorResponse: false,
      });

      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.status === 404) {
          throw new ChatError(ChatErrorType.ENDPOINT_NOT_FOUND, {
            status: error.status,
            url: apiUrl,
          });
        }

        console.warn(
          `Retry attempt ${maxRetries - retries + 1}: Server responded with ${error.status}`,
        );
        retries -= 1;

        if (retries <= 0) {
          if (error.status >= 500) {
            throw new ChatError(ChatErrorType.SERVICE, { status: error.status });
          }
          throw new ChatError(ChatErrorType.UNKNOWN, {
            status: error.status,
            message: `Server responded with ${error.status} after multiple attempts`,
          });
        }
      } else {
        console.error(`Fetch error (attempt ${maxRetries - retries + 1}):`, error);
        retries -= 1;

        if (retries <= 0) {
          if (error instanceof ChatError) {
            throw error;
          }
          if (error instanceof TypeError || (error as Error).message.includes('network')) {
            throw new ChatError(ChatErrorType.NETWORK, error);
          }
          throw new ChatError(ChatErrorType.UNKNOWN, error);
        }
      }

      await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** (maxRetries - retries)));
    }
  }
};

export const processApiResponse = async (response: Response): Promise<string> => {
  const responseClone = response.clone();

  try {
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      const data: { content?: string } = await response.json();
      if (data?.content) {
        return formatText(data.content);
      }

      console.error('Invalid JSON response format:', data);
      return DEFAULT_INSTRUCTIONS;
    }

    const textData = await response.text();
    console.error('Response is not JSON:', textData.substring(0, 200));
    return DEFAULT_INSTRUCTIONS;
  } catch (error) {
    console.error('Failed to parse API response:', error);

    try {
      const textData = await responseClone.text();
      console.error('Response content:', textData.substring(0, 200));
      return DEFAULT_INSTRUCTIONS;
    } catch (textError) {
      console.error('Failed to read response as text:', textError);
      throw new Error(`Failed to process response: ${(error as Error).message}`);
    }
  }
};

let memoryCache: string | null = null;
let memoryCacheTimestamp = 0;
let isInitializing = false;
let initializationPromise: Promise<string> | null = null;

export const fetchTravelInstructions = async (): Promise<string> => {
  if (isInitializing && initializationPromise) {
    return initializationPromise;
  }

  if (memoryCache && Date.now() - memoryCacheTimestamp < CACHE_CONFIG.CACHE_DURATION) {
    return memoryCache;
  }

  try {
    isInitializing = true;
    initializationPromise = (async () => {
      const cachedData = await getCachedData();
      if (cachedData) {
        memoryCache = cachedData;
        memoryCacheTimestamp = Date.now();
        return cachedData;
      }

      console.log('Fetching fresh travel instructions...');
      const apiUrl = '/api/travel-instructions';
      console.log(`Using travel instructions API URL: ${apiUrl}`);

      try {
        const response = await fetchWithRetry(apiUrl);
        const instructions = await processApiResponse(response);
        memoryCache = instructions;
        memoryCacheTimestamp = Date.now();
        await setCachedData(instructions);
        return instructions;
      } catch (error) {
        console.error('Error fetching from API:', error);

        if (error instanceof ChatError && error.type === ChatErrorType.ENDPOINT_NOT_FOUND) {
          console.warn(
            'Travel instructions API endpoint not found. Ensure the server is running on port 3003',
          );
        }

        if (error instanceof ChatError) {
          const { title, message, suggestion } = error.getErrorMessage();
          console.error(`${title}: ${message}\n${suggestion}`);
        }

        return DEFAULT_INSTRUCTIONS;
      }
    })();

    return await initializationPromise;
  } catch (error) {
    console.error('Error fetching travel instructions:', error);

    if (memoryCache) {
      console.log('Using memory cache as fallback due to error');
      return memoryCache;
    }

    console.log('Using default travel instructions as fallback');
    return DEFAULT_INSTRUCTIONS;
  } finally {
    isInitializing = false;
    initializationPromise = null;
  }
};

export const resetTravelInstructionsCache = (): void => {
  memoryCache = null;
  memoryCacheTimestamp = 0;
  isInitializing = false;
  initializationPromise = null;
};
