import type { FollowUpQuestion } from '@/types/chat';

interface FollowUpItem {
  question?: string;
  id?: string;
  reference?: string;
  title?: string;
  category?: FollowUpQuestion['category'];
  icon?: string;
  confidence?: number;
  groundingScore?: number;
  sourceGrounding?: string;
}

export const mapFollowUpQuestions = (
  messageId: string,
  items: Array<string | FollowUpItem> = [],
): FollowUpQuestion[] =>
  items
    .filter(Boolean)
    .map((item, index) => {
      const isString = typeof item === 'string';
      const question = isString ? item : item.question;
      if (!question) return null;

      const baseId =
        (isString ? undefined : item.id || item.reference || item.title) ??
        `${messageId}-fu-${index}`;

      return {
        id: baseId,
        question,
        category: (isString ? 'general' : item.category) || 'general',
        icon: isString ? undefined : item.icon,
        confidence: isString ? undefined : item.confidence,
        groundingScore: isString ? undefined : item.groundingScore,
        sourceGrounding: isString ? undefined : item.sourceGrounding,
      } as FollowUpQuestion;
    })
    .filter((q): q is FollowUpQuestion => q !== null);

export const areFollowUpQuestionsEqual = (
  prevQuestions?: FollowUpQuestion[],
  nextQuestions?: FollowUpQuestion[],
) => {
  if (prevQuestions === nextQuestions) {
    return true;
  }
  if (!prevQuestions || !nextQuestions) {
    return !prevQuestions && !nextQuestions;
  }
  if (prevQuestions.length !== nextQuestions.length) {
    return false;
  }

  return prevQuestions.every((prevQuestion, index) => {
    const nextQuestion = nextQuestions[index];
    if (!nextQuestion) {
      return false;
    }
    return (
      prevQuestion.id === nextQuestion.id &&
      prevQuestion.question === nextQuestion.question &&
      prevQuestion.category === nextQuestion.category &&
      prevQuestion.icon === nextQuestion.icon &&
      prevQuestion.confidence === nextQuestion.confidence &&
      prevQuestion.groundingScore === nextQuestion.groundingScore &&
      prevQuestion.sourceGrounding === nextQuestion.sourceGrounding
    );
  });
};
