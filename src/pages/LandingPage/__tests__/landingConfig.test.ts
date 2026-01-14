import { describe, expect, it } from 'vitest';
import { landingFeatures, quickAskPrompts } from '../landingConfig';

describe('landingConfig', () => {
  it('exposes quick ask prompts with labels and queries', () => {
    expect(quickAskPrompts.length).toBeGreaterThanOrEqual(4);
    quickAskPrompts.forEach((prompt) => {
      expect(prompt.label).toBeTypeOf('string');
      expect(prompt.query).toBeTypeOf('string');
      expect(prompt.query.length).toBeGreaterThan(10);
    });
  });

  it('includes a SCIP portal feature flagged as action', () => {
    const feature = landingFeatures.find((item) => item.id === 'scipPortal');
    expect(feature).toBeDefined();
    expect(feature?.kind).toBe('action');
    expect(feature?.to).toBeUndefined();
  });

  it('marks resources as link with review badge', () => {
    const resources = landingFeatures.find((item) => item.id === 'resources');
    expect(resources).toBeDefined();
    expect(resources?.kind).toBe('link');
    expect(resources?.badge).toBe('Under Review');
  });

  it('includes at least one navigable feature link', () => {
    const linkFeatures = landingFeatures.filter((item) => item.kind === 'link');
    expect(linkFeatures.length).toBeGreaterThanOrEqual(2);
    linkFeatures.forEach((feature) => {
      expect(feature.to).toBeDefined();
    });
  });
});
