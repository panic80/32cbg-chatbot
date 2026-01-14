export const MAINTENANCE_MODE = import.meta.env.VITE_MAINTENANCE_MODE === 'true';
export const MAINTENANCE_MESSAGE =
  import.meta.env.VITE_MAINTENANCE_MESSAGE ||
  'System maintenance in progress. Chat functionality is temporarily disabled.';
