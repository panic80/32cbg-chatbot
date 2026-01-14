import { useState, useRef, useEffect, useCallback } from 'react';
import type { KeyboardEvent, RefObject } from 'react';
import { apiClient, ApiError } from '@/api/client';

interface Prediction {
  description: string;
  place_id: string;
  structured_formatting?: {
    main_text: string;
    secondary_text: string;
  };
}

interface UsePlaceAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  countryRestriction?: string;
  minInputLength?: number;
  debounceMs?: number;
}

interface UsePlaceAutocompleteResult {
  inputRef: RefObject<HTMLInputElement>;
  dropdownRef: RefObject<HTMLDivElement>;
  predictions: Prediction[];
  isLoading: boolean;
  error: string | null;
  showDropdown: boolean;
  selectedIndex: number;
  handleInputChange: (nextValue: string) => void;
  handleInputFocus: () => void;
  handleKeyDown: (event: KeyboardEvent<HTMLInputElement>) => void;
  handlePredictionClick: (prediction: Prediction) => void;
  handlePredictionHover: (index: number) => void;
}

const createSessionToken = () => Math.random().toString(36).substring(2) + Date.now().toString(36);

export const usePlaceAutocomplete = ({
  value,
  onChange,
  countryRestriction = 'ca',
  minInputLength = 2,
  debounceMs = 300,
}: UsePlaceAutocompleteProps): UsePlaceAutocompleteResult => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const hasPredictions = predictions.length > 0;

  const sessionTokenRef = useRef<string>(createSessionToken());
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearDebounce = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
  }, []);

  const resetDropdownState = useCallback(() => {
    setShowDropdown(false);
    setSelectedIndex(-1);
  }, []);

  const fetchPredictions = useCallback(
    async (input: string) => {
      if (!input || input.length < minInputLength) {
        setPredictions([]);
        resetDropdownState();
        setError(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams({
          input,
          sessiontoken: sessionTokenRef.current,
          components: `country:${countryRestriction}`,
        });

        const data = await apiClient.getJson<{
          status: string;
          predictions: Prediction[];
        }>(`/api/maps/autocomplete?${params}`, {
          parseErrorResponse: false,
        });

        if (data.status && data.status !== 'OK') {
          setPredictions([]);
          resetDropdownState();
          setError('Location search unavailable');
          return;
        }

        if (Array.isArray(data.predictions)) {
          setPredictions(data.predictions);
          setShowDropdown(true);
        }
      } catch (err) {
        if (err instanceof ApiError) {
          console.error('Error fetching predictions:', err.status, err.statusText);
        } else {
          console.error('Error fetching predictions:', err);
        }
        setPredictions([]);
        resetDropdownState();
        setError('Location search unavailable');
      } finally {
        setIsLoading(false);
      }
    },
    [countryRestriction, minInputLength, resetDropdownState],
  );

  const selectPrediction = useCallback(
    async (prediction: Prediction) => {
      clearDebounce();

      try {
        const params = new URLSearchParams({
          place_id: prediction.place_id,
          sessiontoken: sessionTokenRef.current,
        });

        const data = await apiClient.getJson<{
          result?: { formatted_address: string };
        }>(`/api/maps/place-details?${params}`, {
          parseErrorResponse: false,
        });

        if (data.result?.formatted_address) {
          onChange(data.result.formatted_address);
        } else {
          onChange(prediction.description);
        }
      } catch (err) {
        if (err instanceof ApiError) {
          console.error('Error fetching place details:', err.status, err.statusText);
        } else {
          console.error('Error fetching place details:', err);
        }
        onChange(prediction.description);
      }

      setPredictions([]);
      resetDropdownState();
      setError(null);
    },
    [clearDebounce, onChange, resetDropdownState],
  );

  const scheduleFetch = useCallback(
    (inputValue: string) => {
      clearDebounce();

      if (!inputValue || inputValue.length < minInputLength) {
        setPredictions([]);
        resetDropdownState();
        return;
      }

      debounceRef.current = setTimeout(() => {
        fetchPredictions(inputValue);
      }, debounceMs);
    },
    [clearDebounce, debounceMs, fetchPredictions, minInputLength, resetDropdownState],
  );

  const handleInputChange = useCallback(
    (nextValue: string) => {
      onChange(nextValue);
      setError(null);
      setSelectedIndex(-1);
      scheduleFetch(nextValue);
    },
    [onChange, scheduleFetch],
  );

  const handleInputFocus = useCallback(() => {
    if (hasPredictions) {
      setShowDropdown(true);
    }
  }, [hasPredictions]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (!showDropdown || predictions.length === 0) {
        return;
      }

      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          setSelectedIndex((prev) => (prev < predictions.length - 1 ? prev + 1 : prev));
          break;
        case 'ArrowUp':
          event.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
          break;
        case 'Enter':
          event.preventDefault();
          if (selectedIndex >= 0 && selectedIndex < predictions.length) {
            selectPrediction(predictions[selectedIndex]);
          }
          break;
        case 'Escape':
          resetDropdownState();
          break;
        default:
          break;
      }
    },
    [predictions, resetDropdownState, selectPrediction, selectedIndex, showDropdown],
  );

  const handlePredictionHover = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const handlePredictionClick = useCallback(
    (prediction: Prediction) => {
      selectPrediction(prediction);
    },
    [selectPrediction],
  );

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        resetDropdownState();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [resetDropdownState]);

  useEffect(
    () => () => {
      clearDebounce();
    },
    [clearDebounce],
  );

  useEffect(() => {
    if (!value || value.length < minInputLength) {
      setPredictions([]);
      resetDropdownState();
    }
  }, [minInputLength, resetDropdownState, value]);

  return {
    inputRef,
    dropdownRef,
    predictions,
    isLoading,
    error,
    showDropdown,
    selectedIndex,
    handleInputChange,
    handleInputFocus,
    handleKeyDown,
    handlePredictionClick,
    handlePredictionHover,
  };
};

export type { Prediction };
