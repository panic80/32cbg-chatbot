import { describe, expect, it } from 'vitest';
import { inferJurisdiction, buildTripPlannerHints } from '../tripPlannerService.js';

describe('tripPlannerService', () => {
  describe('inferJurisdiction', () => {
    it('infers Ontario from ON', () => {
      expect(inferJurisdiction('I am traveling to ON')).toBe('Ontario, Canada');
    });

    it('infers Quebec from Québec', () => {
      expect(inferJurisdiction('Traveling to Québec next week')).toBe('Quebec, Canada');
    });

    it('infers British Columbia from BC', () => {
      expect(inferJurisdiction('BC travel instructions')).toBe('British Columbia, Canada');
    });

    it('returns undefined for unknown locations', () => {
      expect(inferJurisdiction('I am going home')).toBeUndefined();
    });

    it('handles non-string input gracefully', () => {
      expect(inferJurisdiction(null as unknown as string)).toBeUndefined();
    });
  });

  describe('buildTripPlannerHints', () => {
    it('builds hints for a specific jurisdiction', () => {
      const hints = buildTripPlannerHints('Alberta, Canada');
      expect(hints).toContain(
        'Alberta private vehicle kilometric rate cents per kilometre Appendix B',
      );
      expect(hints).toContain('meal allowance rates Alberta');
      expect(hints).toContain('incidental allowance daily rate');
    });

    it('defaults to Ontario if no jurisdiction is provided', () => {
      const hints = buildTripPlannerHints(undefined);
      expect(hints).toContain(
        'Ontario private vehicle kilometric rate cents per kilometre Appendix B',
      );
      expect(hints).toContain('meal allowance rates Ontario');
    });
  });
});
