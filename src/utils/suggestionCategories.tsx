import React from 'react';
import type { LucideIcon } from 'lucide-react';
import { ArrowRight, HelpCircle, Lightbulb, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { FollowUpQuestion } from '@/types';

type SuggestionCategory = NonNullable<FollowUpQuestion['category']>;
type SuggestionCategoryKey = SuggestionCategory | 'default';

interface SuggestionCategoryConfig {
  label: string;
  icon: LucideIcon;
  iconClasses: Record<'card' | 'chip', string>;
}

const CATEGORY_CONFIG: Record<SuggestionCategoryKey, SuggestionCategoryConfig> = {
  clarification: {
    label: 'Clarification',
    icon: HelpCircle,
    iconClasses: {
      card: 'text-blue-500 dark:text-blue-400',
      chip: 'text-blue-600 dark:text-blue-400',
    },
  },
  related: {
    label: 'Related Topic',
    icon: Search,
    iconClasses: {
      card: 'text-green-500 dark:text-green-400',
      chip: 'text-green-600 dark:text-green-400',
    },
  },
  practical: {
    label: 'Practical',
    icon: Lightbulb,
    iconClasses: {
      card: 'text-orange-500 dark:text-orange-400',
      chip: 'text-orange-600 dark:text-orange-400',
    },
  },
  explore: {
    label: 'Explore More',
    icon: ArrowRight,
    iconClasses: {
      card: 'text-purple-500 dark:text-purple-400',
      chip: 'text-purple-600 dark:text-purple-400',
    },
  },
  general: {
    label: 'General',
    icon: HelpCircle,
    iconClasses: {
      card: 'text-gray-500 dark:text-gray-400',
      chip: 'text-gray-600 dark:text-gray-400',
    },
  },
  default: {
    label: 'Question',
    icon: HelpCircle,
    iconClasses: {
      card: 'text-gray-500 dark:text-gray-400',
      chip: 'text-gray-600 dark:text-gray-400',
    },
  },
};

const DEFAULT_CATEGORY_KEY: SuggestionCategoryKey = 'default';

const hasCategoryConfig = (category: string): category is SuggestionCategoryKey =>
  Object.prototype.hasOwnProperty.call(CATEGORY_CONFIG, category);

const resolveCategoryKey = (category?: FollowUpQuestion['category']): SuggestionCategoryKey => {
  if (category && hasCategoryConfig(category)) {
    return category;
  }

  return DEFAULT_CATEGORY_KEY;
};

export const getSuggestionCategoryLabel = (category?: FollowUpQuestion['category']): string => {
  const key = resolveCategoryKey(category);
  return CATEGORY_CONFIG[key].label;
};

export interface SuggestionCategoryIconOptions {
  size?: number;
  variant?: 'card' | 'chip';
  className?: string;
}

export const getSuggestionCategoryIcon = (
  category?: FollowUpQuestion['category'],
  { size = 14, variant = 'card', className }: SuggestionCategoryIconOptions = {},
): React.ReactNode => {
  const key = resolveCategoryKey(category);
  const { icon: Icon, iconClasses } = CATEGORY_CONFIG[key];
  const variantClass = iconClasses[variant] ?? '';

  return <Icon size={size} className={cn(variantClass, className)} />;
};
