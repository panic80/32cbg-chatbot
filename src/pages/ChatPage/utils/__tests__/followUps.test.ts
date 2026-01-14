import { describe, expect, it } from 'vitest';
import { areFollowUpQuestionsEqual, mapFollowUpQuestions } from '../followUps';

describe('mapFollowUpQuestions', () => {
  it('maps structured follow up objects', () => {
    const result = mapFollowUpQuestions('msg-1', [
      { question: 'What is next?', category: 'actions', icon: 'sparkles' },
    ]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      id: 'msg-1-fu-0',
      question: 'What is next?',
      category: 'actions',
      icon: 'sparkles',
    });
  });

  it('filters invalid entries and handles strings', () => {
    const result = mapFollowUpQuestions('msg-2', ['Plain string', null]);

    expect(result).toHaveLength(1);
    expect(result[0]?.question).toBe('Plain string');
  });
});

describe('areFollowUpQuestionsEqual', () => {
  it('compares arrays deeply', () => {
    const first = mapFollowUpQuestions('msg', [{ question: 'Q1' }]);
    const second = mapFollowUpQuestions('msg', [{ question: 'Q1' }]);

    expect(areFollowUpQuestionsEqual(first, second)).toBe(true);
  });

  it('detects differences in order or content', () => {
    const first = mapFollowUpQuestions('msg', [{ question: 'Q1' }, { question: 'Q2' }]);
    const second = mapFollowUpQuestions('msg', [{ question: 'Q1' }, { question: 'Q3' }]);

    expect(areFollowUpQuestionsEqual(first, second)).toBe(false);
  });
});
