// Original suggestions (kept for backward compatibility)
export const WELCOME_SUGGESTIONS = [
  {
    title: 'Document Requirements for TD in Simple Steps',
    subtitle: 'explain travel claim process',
    icon: '📋',
  },
  { title: 'LTA eligibility', subtitle: 'leave travel assistance benefits', icon: '🏖️' },
  { title: 'Car Storage', subtitle: 'vehicle storage policies', icon: '🚗' },
  { title: 'Travel authorization', subtitle: 'approval process and forms', icon: '✈️' },
  { title: 'POMV vs GMT', subtitle: 'pros and cons comparison', icon: '🚙' },
  { title: 'Can I choose to drive?', subtitle: 'personal vehicle options', icon: '🚘' },
];

// New categorized suggestions structure
export interface SuggestionItem {
  title: string;
  subtitle: string;
  icon: string;
}

export interface CategoryData {
  id: string;
  label: string;
  /** Optional shorter label for compact views (e.g., mobile tabs) */
  shortLabel?: string;
  icon: string;
  questions: SuggestionItem[];
}

export const CATEGORIZED_SUGGESTIONS: CategoryData[] = [
  {
    id: 'popular',
    label: 'Popular',
    icon: '⭐',
    questions: [
      {
        title: 'Document Requirements for TD in Simple Steps',
        subtitle: 'explain travel claim process',
        icon: '📋',
      },
      { title: 'Meal allowances', subtitle: 'daily rates and eligibility', icon: '🍽️' },
      { title: 'POMV vs GMT', subtitle: 'pros and cons comparison', icon: '🚙' },
      { title: 'Travel authorization', subtitle: 'approval process and forms', icon: '✈️' },
      { title: 'Can I choose to drive?', subtitle: 'personal vehicle options', icon: '🚘' },
      {
        title: 'Can I choose to drive instead of GMT, in 5 points',
        subtitle: 'personal vehicle options overview',
        icon: '🚘',
      },
      {
        title: 'How do i book Hotels for TD?',
        subtitle: 'lodging options and booking rules',
        icon: '🏨',
      },
    ],
  },
  {
    id: 'travel',
    label: 'Travel & Claims',
    icon: '✈️',
    questions: [
      {
        title: 'International travel',
        subtitle: 'procedures for travel outside Canada',
        icon: '🌍',
      },
      { title: 'Travel advances', subtitle: 'how to request travel funds', icon: '💵' },
      {
        title: 'How do i book Hotels for TD?',
        subtitle: 'lodging options and booking rules',
        icon: '🏨',
      },
      { title: 'Can I choose to drive?', subtitle: 'personal vehicle options', icon: '🚘' },
      { title: 'Travel card usage', subtitle: 'government credit card policies', icon: '💳' },
    ],
  },
  {
    id: 'benefits',
    label: 'Benefits & Allowances',
    icon: '💰',
    questions: [
      { title: 'LTA eligibility', subtitle: 'leave travel assistance benefits', icon: '🏖️' },
      { title: 'Relocation benefits', subtitle: 'entitlements during posting', icon: '📦' },
      { title: 'Foreign service', subtitle: 'benefits for overseas postings', icon: '🌐' },
      { title: 'Incidental expenses', subtitle: "what's covered under incidentals", icon: '💸' },
    ],
  },
  {
    id: 'administration',
    label: 'Administration',
    shortLabel: 'Admin',
    icon: '📑',
    questions: [
      { title: 'Car Storage', subtitle: 'vehicle storage policies', icon: '🚗' },
      { title: 'Expense claims', subtitle: 'submission and approval process', icon: '📄' },
      {
        title: 'Document Requirements for TD in Simple Steps',
        subtitle: 'required forms and receipts',
        icon: '📋',
      },
      { title: 'Approval process', subtitle: 'who approves what and when', icon: '✅' },
    ],
  },
];

// Helper function to get all questions from all categories
export const getAllCategorizedQuestions = (): SuggestionItem[] => {
  return CATEGORIZED_SUGGESTIONS.flatMap((category) => category.questions);
};
