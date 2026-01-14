import { differenceInDays, format } from 'date-fns';
import type { TripData, DistanceData, MealEntitlementDay, MileageContext } from '../types';
import {
  INCIDENT_ALLOWANCE_REDUCED_RATE,
  INCIDENT_ALLOWANCE_STANDARD_DAYS,
  INCIDENT_ALLOWANCE_STANDARD_RATE,
} from '@/constants/travel';
import { getTransportLabel } from '../constants';

const currencyFormatter = new Intl.NumberFormat('en-CA', {
  style: 'currency',
  currency: 'CAD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export const formatCurrency = (value: number) => currencyFormatter.format(value);

export const formatKilometres = (value: number) => {
  if (!Number.isFinite(value)) {
    return '';
  }
  return Number.isInteger(value) ? `${value.toFixed(0)} km` : `${value.toFixed(1)} km`;
};

export const calculateTripDurationInDays = (departure?: Date, returnDate?: Date) => {
  if (!departure || !returnDate) {
    return null;
  }
  return differenceInDays(returnDate, departure) + 1;
};

export const calculateIncidentalCost = (tripDuration: number | null) => {
  if (!tripDuration || tripDuration <= 0) {
    return null;
  }

  if (tripDuration <= INCIDENT_ALLOWANCE_STANDARD_DAYS) {
    return tripDuration * INCIDENT_ALLOWANCE_STANDARD_RATE;
  }

  if (tripDuration === INCIDENT_ALLOWANCE_STANDARD_DAYS + 1) {
    return (INCIDENT_ALLOWANCE_STANDARD_DAYS + 1) * INCIDENT_ALLOWANCE_STANDARD_RATE;
  }

  const intermediateDays = tripDuration - (INCIDENT_ALLOWANCE_STANDARD_DAYS + 1);
  const standardDaysCost = INCIDENT_ALLOWANCE_STANDARD_DAYS * INCIDENT_ALLOWANCE_STANDARD_RATE;
  const reducedDaysCost = Math.max(intermediateDays, 0) * INCIDENT_ALLOWANCE_REDUCED_RATE;
  const finalDayCost = INCIDENT_ALLOWANCE_STANDARD_RATE;

  return standardDaysCost + reducedDaysCost + finalDayCost;
};

export const calculateMealEntitlements = (
  tripDuration: number | null,
  rnqProvided: boolean,
): MealEntitlementDay[] => {
  if (!tripDuration || tripDuration <= 0) {
    return [];
  }

  if (tripDuration === 1) {
    return [
      {
        dayLabel: 'Day 1 (Same-day travel)',
        meals: ['Lunch'],
      },
    ];
  }

  const entitlements: MealEntitlementDay[] = [
    {
      dayLabel: 'Day 1 (Departure)',
      meals: ['Lunch', 'Dinner'],
    },
  ];

  if (!rnqProvided && tripDuration > 2) {
    for (let day = 2; day <= tripDuration - 1; day += 1) {
      entitlements.push({
        dayLabel: `Day ${day}`,
        meals: ['Breakfast', 'Lunch', 'Dinner'],
      });
    }
  }

  entitlements.push({
    dayLabel: `Day ${tripDuration} (Return)`,
    meals: ['Breakfast', 'Lunch'],
  });

  if (rnqProvided) {
    return entitlements.filter((entry, index) => index === 0 || index === entitlements.length - 1);
  }

  return entitlements;
};

const PROVINCE_MAP: Record<string, string> = {
  ab: 'Alberta',
  alberta: 'Alberta',
  bc: 'British Columbia',
  'british columbia': 'British Columbia',
  mb: 'Manitoba',
  manitoba: 'Manitoba',
  nb: 'New Brunswick',
  'new brunswick': 'New Brunswick',
  nl: 'Newfoundland and Labrador',
  'newfoundland and labrador': 'Newfoundland and Labrador',
  nfld: 'Newfoundland and Labrador',
  "st. john's": 'Newfoundland and Labrador',
  ns: 'Nova Scotia',
  'nova scotia': 'Nova Scotia',
  nt: 'Northwest Territories',
  'northwest territories': 'Northwest Territories',
  'northwest territory': 'Northwest Territories',
  nu: 'Nunavut',
  nunavut: 'Nunavut',
  on: 'Ontario',
  ont: 'Ontario',
  ontario: 'Ontario',
  pei: 'Prince Edward Island',
  'prince edward island': 'Prince Edward Island',
  qc: 'Quebec',
  québec: 'Quebec',
  quebec: 'Quebec',
  sk: 'Saskatchewan',
  saskatchewan: 'Saskatchewan',
  yt: 'Yukon',
  yukon: 'Yukon',
};

export const extractProvince = (location: string | undefined) => {
  if (!location) return null;

  const parts = location
    .split(',')
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);

  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const normalized = parts[i];
    if (PROVINCE_MAP[normalized]) {
      return PROVINCE_MAP[normalized];
    }

    const tokens = normalized.split(/[\s-]+/).map((token) => token.replace(/[^a-z]/g, ''));
    for (const token of tokens) {
      if (token && PROVINCE_MAP[token]) {
        return PROVINCE_MAP[token];
      }
    }
  }

  return null;
};

export const buildMileageContext = (
  data: TripData,
  distance: DistanceData | null,
): MileageContext => {
  const departureProvince = extractProvince(data.departureLocation);
  const destinationProvince = extractProvince(data.arrivalLocation);
  const locationLabel = departureProvince || destinationProvince || 'departure region';
  const distanceValue = distance?.distance.value;
  const distanceKm = typeof distanceValue === 'number' ? distanceValue / 1000 : null;
  const roundTripKm = distanceKm !== null ? distanceKm * 2 : null;
  const distanceText = distance?.distance.text ?? null;
  const roundTripDistanceText =
    roundTripKm !== null
      ? formatKilometres(roundTripKm)
      : distanceText
        ? `2 × ${distanceText}`
        : null;

  return {
    locationLabel,
    locationLabelLower: locationLabel.toLowerCase(),
    distanceText,
    roundTripDistanceText,
    distanceKm,
    roundTripKm,
  };
};

export const buildCostEstimateSection = (
  data: TripData,
  distance: DistanceData | null,
  tripDuration: number | null,
  mealEntitlements: MealEntitlementDay[],
) => {
  const incidentalCost = calculateIncidentalCost(tripDuration);

  const lines: string[] = [];
  const costRows: Array<{ label: string; value: string }> = [];
  const internalNotes: string[] = [];

  if (incidentalCost !== null) {
    const durationLabel = tripDuration === 1 ? '1 day' : `${tripDuration} days`;
    lines.push(`• Incidentals (${durationLabel}): ${formatCurrency(incidentalCost)}`);
    costRows.push({
      label: `Incidentals (${durationLabel})`,
      value: formatCurrency(incidentalCost),
    });
  } else {
    costRows.push({
      label: 'Incidentals',
      value: 'No entitlement calculated',
    });
  }

  if (mealEntitlements.length) {
    const mealSummary = mealEntitlements
      .map((entry) => `${entry.dayLabel}: ${entry.meals.join(', ')}`)
      .join('; ');
    const rnqMessage = data.rnqProvided ? ' (travel days only — R&Q provided)' : '';
    lines.push(
      `• Meals: ${mealSummary}${rnqMessage}. Retrieve destination meal per diems (breakfast/lunch/dinner) and multiply accordingly.`,
    );
    costRows.push({
      label: 'Meals',
      value: `${mealSummary}${data.rnqProvided ? ' (travel days only)' : ''}`,
    });
    internalNotes.push(
      'Lookup current meal per diem table for the destination and calculate eligible meals.',
    );
  } else {
    lines.push('• Meals: Not entitled based on travel timing and provided R&Q coverage.');
    costRows.push({
      label: 'Meals',
      value: 'No entitlement (timing/R&Q).',
    });
  }

  const routeDescription = `${data.departureLocation || 'departure location'} → ${
    data.arrivalLocation || 'arrival location'
  }`;
  const transportLabel = getTransportLabel(data.transportMethod);
  const mileageContext = buildMileageContext(data, distance);
  const { locationLabel, locationLabelLower, distanceText, roundTripDistanceText } = mileageContext;
  const roundTripDisplay = roundTripDistanceText ?? 'round-trip distance';

  if (data.transportMethod === 'personal-vehicle') {
    if (distance?.distance.text && distance.distance.value) {
      lines.push(
        `• Mileage: Retrieve the current private-vehicle kilometric rate for ${locationLabel} (cents per kilometre) and apply it to ${distanceText} each way (${roundTripDisplay} total) to determine the mileage cost.`,
      );
      costRows.push({
        label: 'Mileage',
        value: `${roundTripDisplay} × current ${locationLabel} POMV rate`,
      });
      internalNotes.push(
        `Query the policy for the ${locationLabel} private-vehicle kilometric rate and multiply it by ${roundTripDisplay} (round trip), converting to dollars if the rate is quoted in cents per kilometre.`,
      );
      internalNotes.push(
        `Explicit retrieval hint: "${locationLabelLower} private vehicle kilometric rate" (express in cents per kilometre).`,
      );
    } else {
      lines.push(
        `• Mileage: Await confirmed distance for ${routeDescription} and then apply the current ${locationLabel} POMV rate (cents per kilometre) to both legs of travel.`,
      );
      costRows.push({
        label: 'Mileage',
        value: 'Awaiting confirmed distance for round-trip POMV calculation',
      });
      internalNotes.push(
        `Fetch the ${locationLabel} private-vehicle kilometric rate once the travel distance is confirmed, then multiply by the full round-trip distance (convert units as required).`,
      );
    }
  } else {
    lines.push(`• Mileage: Not entitled when travelling by ${transportLabel}.`);
    costRows.push({
      label: 'Mileage',
      value: `Not entitled (${transportLabel} is Crown-provided transport).`,
    });
  }

  if (!lines.length) {
    return '';
  }

  let section = `
💵 **Estimated Costs:**
${lines.join('\n')}
`;

  section += `**Please combine the RAG-derived kilometric mileage cost with the incidentals above to present the total trip estimate.**
`;

  if (costRows.length) {
    const tableRows = costRows.map((row) => `| ${row.label} | ${row.value} |`).join('\n');
    section += `
| Component | Estimated Cost / Notes |
| --- | --- |
${tableRows}
`;
  }

  if (internalNotes.length) {
    const noteLines = internalNotes.map((note) => `- ${note}`).join('\n');
    section += `
<details>
<summary>Internal calculation notes</summary>
${noteLines}
</details>
`;
  }

  return section;
};

export const describeTripBasics = (data: TripData) => {
  const transport = getTransportLabel(data.transportMethod);
  const departure = data.departureDate
    ? format(data.departureDate, 'MMMM dd, yyyy')
    : 'Not specified';
  const returnDate = data.returnDate ? format(data.returnDate, 'MMMM dd, yyyy') : 'Not specified';
  return { transport, departure, returnDate };
};
