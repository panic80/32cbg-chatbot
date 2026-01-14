export const FOLLOW_UP_CATEGORIES = {
  CLARIFICATION: 'clarification',
  RELATED: 'related',
  PRACTICAL: 'practical',
  EXPLORE: 'explore',
  GENERAL: 'general',
} as const;

export const DEFAULT_CONFIDENCE = {
  HIGH: 0.9,
  MEDIUM: 0.7,
  LOW: 0.6,
  FALLBACK: 0.5,
  MINIMAL: 0.4,
};

export const FALLBACK_QUESTIONS = {
  TRAVEL: 'What documentation is required for this type of travel?',
  CLAIM: 'What is the timeline for submitting this claim?',
  ALLOWANCE: 'Are there any restrictions or conditions for this allowance?',
  GENERIC_EXAMPLES: 'Can you provide more specific examples?',
  GENERIC_NEXT_STEPS: 'What are the next steps I should take?',
};
