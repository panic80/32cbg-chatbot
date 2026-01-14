/**
 * Trip Planner Service.
 * Contains logic for inferring jurisdiction and building search hints for travel planning.
 */

interface ProvinceMatcher {
  name: string;
  re: RegExp;
}

const PROVINCE_MATCHERS: ProvinceMatcher[] = [
  { name: 'Alberta', re: /\b(AB|Alberta)\b/i },
  { name: 'British Columbia', re: /\b(BC|British\s+Columbia)\b/i },
  { name: 'Manitoba', re: /\b(MB|Manitoba)\b/i },
  { name: 'New Brunswick', re: /\b(NB|New\s+Brunswick)\b/i },
  { name: 'Newfoundland and Labrador', re: /\b(NL|Newfoundland(?:\s+and\s+Labrador)?|Nfld)\b/i },
  { name: 'Nova Scotia', re: /\b(NS|Nova\s+Scotia)\b/i },
  { name: 'Ontario', re: /\b(ON|Ont|Ontario)\b/i },
  { name: 'Prince Edward Island', re: /\b(PE|PEI|Prince\s+Edward\s+Island)\b/i },
  { name: 'Quebec', re: /\b(QC|Quebec|Québec)\b/i },
  { name: 'Saskatchewan', re: /\b(SK|Saskatchewan)\b/i },
  { name: 'Yukon', re: /\b(YT|Yukon)\b/i },
  { name: 'Northwest Territories', re: /\b(NT|NWT|Northwest\s+Territories?)\b/i },
  { name: 'Nunavut', re: /\b(NU|Nunavut)\b/i },
];

/**
 * Infer the jurisdiction (province) from a user message.
 * @param {string} message - User message
 * @returns {string|undefined} Jurisdiction string or undefined
 */
export const inferJurisdiction = (message: unknown): string | undefined => {
  if (typeof message !== 'string') return undefined;
  const found = PROVINCE_MATCHERS.find((p) => p.re.test(message));
  return found ? `${found.name}, Canada` : undefined;
};

/**
 * Build retrieval focus hints based on the jurisdiction.
 * @param {string} jurisdiction - Jurisdiction string
 * @returns {string[]} Array of hint strings
 */
export const buildTripPlannerHints = (jurisdiction: unknown): string[] => {
  const hints: string[] = [];
  const province = jurisdiction ? String(jurisdiction).split(',')[0] : 'Ontario';
  hints.push(`${province} private vehicle kilometric rate cents per kilometre Appendix B`);
  hints.push(`meal allowance rates ${province}`);
  hints.push(`incidental allowance daily rate`);
  return hints;
};
