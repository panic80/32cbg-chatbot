import React from 'react';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';
import { usePlaceAutocomplete } from '@/hooks/usePlaceAutocomplete';

interface PlaceAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  id?: string;
  name?: string;
  required?: boolean;
  countryRestriction?: string;
}

export const PlaceAutocompleteSimple: React.FC<PlaceAutocompleteProps> = ({
  value,
  onChange,
  placeholder = 'Enter a location',
  className,
  disabled = false,
  id,
  name,
  required = false,
  countryRestriction = 'ca',
}) => {
  const {
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
  } = usePlaceAutocomplete({
    value,
    onChange,
    countryRestriction,
  });

  return (
    <div className="relative">
      <div className="relative">
        <Input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(event) => handleInputChange(event.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={handleInputFocus}
          placeholder={placeholder}
          className={className}
          disabled={disabled}
          id={id}
          name={name}
          required={required}
          autoComplete="off"
        />
        {isLoading && (
          <Loader2 className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
        )}
      </div>
      {error && <div className="mt-1 text-xs text-red-500">{error}</div>}

      {showDropdown && predictions.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 mt-1 w-full rounded-md border bg-white shadow-lg dark:bg-gray-800"
        >
          <ul className="max-h-60 overflow-auto py-1">
            {predictions.map((prediction, index) => (
              <li
                key={prediction.place_id}
                className={cn(
                  'cursor-pointer px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700',
                  selectedIndex === index && 'bg-gray-100 dark:bg-gray-700',
                )}
                onClick={() => handlePredictionClick(prediction)}
                onMouseEnter={() => handlePredictionHover(index)}
              >
                {prediction.structured_formatting ? (
                  <div>
                    <div className="font-medium text-gray-900 dark:text-gray-100">
                      {prediction.structured_formatting.main_text}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {prediction.structured_formatting.secondary_text}
                    </div>
                  </div>
                ) : (
                  <div className="text-gray-900 dark:text-gray-100">{prediction.description}</div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
