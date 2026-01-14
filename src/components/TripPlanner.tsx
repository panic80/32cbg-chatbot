import { useMemo, useState } from 'react';
import { CalendarIcon, MapIcon, CheckCircle2 } from 'lucide-react';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Calendar } from '@/components/ui/calendar';
import { PlaceAutocompleteSimple as PlaceAutocomplete } from '@/components/PlaceAutocompleteSimple';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { DEFAULT_DEPARTURE_TIME, DEFAULT_RETURN_TIME } from '@/constants/travel';
import { useTripPlannerForm } from '@/pages/TripPlanner/hooks/useTripPlannerForm';
import { useDistanceMatrix } from '@/pages/TripPlanner/hooks/useDistanceMatrix';
import { TRANSPORT_METHODS } from '@/pages/TripPlanner/constants';
import type { TripData, DistanceData, MealEntitlementDay } from '@/pages/TripPlanner/types';
import { buildMileageContext, formatKilometres } from '@/pages/TripPlanner/utils/calculations';
import { generateTripPlanMessage } from '@/pages/TripPlanner/utils/generateTripPlanMessage';

interface TripPlannerProps {
  onSubmit: (tripPlan: string) => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const generateMealSummary = (mealEntries: MealEntitlementDay[]) =>
  mealEntries.map((entry) => `${entry.dayLabel}: ${entry.meals.join(', ')}`).join('; ');

const TripSummary = ({
  tripData,
  distanceData,
  tripDuration,
  mealEntitlements,
}: {
  tripData: TripData;
  distanceData: DistanceData | null;
  tripDuration: number | null;
  mealEntitlements: MealEntitlementDay[];
}) => {
  const mileageContext = useMemo(
    () => buildMileageContext(tripData, distanceData),
    [tripData, distanceData],
  );

  return (
    <div className="space-y-3 rounded-lg border border-dashed border-[var(--border)] bg-[var(--background-secondary)] p-4 text-sm">
      <p className="font-medium text-[var(--text)]">Trip Summary</p>
      <ul className="space-y-2 text-[var(--text-secondary)]">
        <li>
          <strong>Transport:</strong>{' '}
          {TRANSPORT_METHODS.find((method) => method.value === tripData.transportMethod)?.label ||
            'Not specified'}
        </li>
        <li>
          <strong>Locations:</strong> {tripData.departureLocation || '—'} →{' '}
          {tripData.arrivalLocation || '—'}
        </li>
        {distanceData && (
          <li>
            <strong>Distance:</strong> {distanceData.distance.text} ({distanceData.duration.text})
          </li>
        )}
        {mileageContext.roundTripDistanceText && (
          <li>
            <strong>Round-trip:</strong> {mileageContext.roundTripDistanceText}
          </li>
        )}
        {tripDuration && (
          <li>
            <strong>Duration:</strong> {tripDuration} day{tripDuration > 1 ? 's' : ''}
          </li>
        )}
        {mealEntitlements.length > 0 && (
          <li>
            <strong>Meals:</strong> {generateMealSummary(mealEntitlements)}
          </li>
        )}
      </ul>
    </div>
  );
};

export const TripPlanner = ({ onSubmit, open: controlledOpen, onOpenChange }: TripPlannerProps) => {
  const [open, setOpen] = useState<boolean>(Boolean(controlledOpen));
  const { tripData, updateTripData, resetTripData, tripDuration, mealEntitlements, isFormValid } =
    useTripPlannerForm();
  const { distanceData, distanceError, isLoadingDistance } = useDistanceMatrix(tripData);

  const handleSubmit = () => {
    const plan = generateTripPlanMessage(tripData, distanceData);
    onSubmit(plan);
    resetTripData();
    setOpen(false);
    onOpenChange?.(false);
  };

  const resolvedOpen = controlledOpen ?? open;
  const setSheetOpen = (nextOpen: boolean) => {
    if (controlledOpen === undefined) {
      setOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  };

  const mileageContext = useMemo(
    () => buildMileageContext(tripData, distanceData),
    [tripData, distanceData],
  );

  return (
    <Sheet open={resolvedOpen} onOpenChange={setSheetOpen}>
      {controlledOpen === undefined && (
        <SheetTrigger
          className="inline-flex h-10 w-10 items-center justify-center rounded-md text-sm font-medium text-[var(--text)] transition-colors hover:bg-[var(--background-secondary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
          title="Trip Planner (Beta)"
        >
          <MapIcon size={18} />
        </SheetTrigger>
      )}
      <SheetContent className="w-[400px] overflow-y-auto sm:w-[540px]">
        <SheetHeader>
          <SheetTitle>Trip Planner (Beta)</SheetTitle>
          <SheetDescription>
            Fill in your travel details to generate a formatted trip plan for your chat.
          </SheetDescription>
        </SheetHeader>

        <div className="space-y-6 py-6">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="transportMethod">Transport Method</Label>
              <Select
                value={tripData.transportMethod}
                onValueChange={(value) =>
                  updateTripData('transportMethod', value as TripData['transportMethod'])
                }
              >
                <SelectTrigger id="transportMethod">
                  <SelectValue placeholder="Select transport method" />
                </SelectTrigger>
                <SelectContent>
                  {TRANSPORT_METHODS.map((method) => (
                    <SelectItem key={method.value || 'none'} value={method.value}>
                      {method.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-1 gap-4">
              <div className="space-y-2">
                <Label htmlFor="departureDate">Departure Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        'w-full justify-start text-left font-normal',
                        !tripData.departureDate && 'text-muted-foreground',
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {tripData.departureDate
                        ? tripData.departureDate.toDateString()
                        : 'Select date'}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={tripData.departureDate}
                      onSelect={(date) => updateTripData('departureDate', date)}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
              </div>

              <div className="space-y-2">
                <Label htmlFor="returnDate">Return Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        'w-full justify-start text-left font-normal',
                        !tripData.returnDate && 'text-muted-foreground',
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {tripData.returnDate ? tripData.returnDate.toDateString() : 'Select date'}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={tripData.returnDate}
                      onSelect={(date) => updateTripData('returnDate', date)}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4">
              <div className="space-y-2">
                <Label htmlFor="departureLocation">Departure Location</Label>
                <PlaceAutocomplete
                  value={tripData.departureLocation}
                  onChange={(value) => updateTripData('departureLocation', value)}
                  placeholder="Enter origin"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="arrivalLocation">Arrival Location</Label>
                <PlaceAutocomplete
                  value={tripData.arrivalLocation}
                  onChange={(value) => updateTripData('arrivalLocation', value)}
                  placeholder="Enter destination"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="purpose">Purpose of Travel</Label>
              <Textarea
                id="purpose"
                value={tripData.purpose}
                onChange={(event) => updateTripData('purpose', event.target.value)}
                placeholder="Provide a concise description of the travel purpose"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="additionalNotes">Additional Notes (Optional)</Label>
              <Textarea
                id="additionalNotes"
                value={tripData.additionalNotes}
                onChange={(event) => updateTripData('additionalNotes', event.target.value)}
                placeholder="Include any extra context or special considerations"
              />
            </div>

            <div className="space-y-3 rounded-lg border border-[var(--border)] p-3">
              <Label className="flex items-center gap-2">
                <Checkbox
                  checked={tripData.rnqProvided}
                  onCheckedChange={(checked) => updateTripData('rnqProvided', Boolean(checked))}
                />
                Rations &amp; Quarters (R&amp;Q) Provided
              </Label>
              <Label className="flex items-center gap-2">
                <Checkbox
                  checked={tripData.travelAuthority}
                  onCheckedChange={(checked) => updateTripData('travelAuthority', Boolean(checked))}
                />
                Travel Authority Obtained
              </Label>
            </div>
          </div>

          {distanceError && (
            <p className="rounded-md bg-red-500/10 p-3 text-sm text-red-500">
              Unable to calculate distance: {distanceError}
            </p>
          )}

          <TripSummary
            tripData={tripData}
            distanceData={distanceData}
            tripDuration={tripDuration}
            mealEntitlements={mealEntitlements}
          />

          <div className="space-y-3 rounded-lg bg-[var(--background-secondary)] p-4 text-sm text-[var(--text-secondary)]">
            <p className="font-medium text-[var(--text)]">Quick Reference</p>
            <ul className="space-y-2">
              <li>
                <strong>Default Depart:</strong> {DEFAULT_DEPARTURE_TIME}
              </li>
              <li>
                <strong>Default Return:</strong> {DEFAULT_RETURN_TIME}
              </li>
              {mileageContext.roundTripDistanceText && (
                <li>
                  <strong>Round Trip:</strong> {mileageContext.roundTripDistanceText}
                </li>
              )}
              {distanceData?.distance?.value && (
                <li>
                  <strong>Single Leg:</strong>{' '}
                  {formatKilometres(distanceData.distance.value / 1000)}
                </li>
              )}
            </ul>
          </div>

          <Button
            className="w-full"
            onClick={handleSubmit}
            disabled={!isFormValid() || isLoadingDistance}
          >
            <CheckCircle2 className="mr-2 h-4 w-4" />
            {isLoadingDistance ? 'Calculating distance…' : 'Generate Trip Plan'}
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export { generateTripPlanMessage } from '@/pages/TripPlanner/utils/generateTripPlanMessage';
export type { TripData, DistanceData } from '@/pages/TripPlanner/types';
