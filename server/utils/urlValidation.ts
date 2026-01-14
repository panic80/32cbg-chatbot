/**
 * URL validation utilities for SSRF protection.
 * Prevents server-side request forgery by validating ingestion URLs.
 */

import dns from 'node:dns/promises';
import net from 'node:net';

/**
 * Check if an IPv4 address is private/internal.
 * @param {string} ip - IPv4 address
 * @returns {boolean} True if private
 */
export const isPrivateIpv4 = (ip: string): boolean => {
  if (typeof ip !== 'string') return false;
  const parts = ip.split('.').map(Number);
  if (parts.length !== 4 || parts.some((segment) => Number.isNaN(segment))) {
    return false;
  }
  if (parts[0] === 10) return true;
  if (parts[0] === 127) return true;
  if (parts[0] === 169 && parts[1] === 254) return true;
  if (parts[0] === 192 && parts[1] === 168) return true;
  if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) return true;
  if (parts[0] === 0) return true;
  return false;
};

/**
 * Check if an IPv6 address is private/internal.
 * @param {string} ip - IPv6 address
 * @returns {boolean} True if private
 */
export const isPrivateIpv6 = (ip: string): boolean => {
  if (typeof ip !== 'string') return false;
  const normalized = ip.toLowerCase();
  if (normalized === '::1') return true;
  if (normalized.startsWith('fc') || normalized.startsWith('fd')) return true; // Unique local
  if (normalized.startsWith('fe80') || normalized.startsWith('fec0')) return true; // Link-local/site-local
  if (normalized === '::') return true;
  return false;
};

/**
 * Resolve hostname to IP addresses.
 * @param {string} hostname - Hostname to resolve
 * @returns {Promise<Array<{address: string, family: number}>>} Resolved addresses
 */
export const resolveHostAddresses = async (
  hostname: string,
): Promise<Array<{ address: string; family: number }>> => {
  try {
    const results = await dns.lookup(hostname, { all: true });
    return results.map(({ address, family }) => ({ address, family }));
  } catch (error) {
    throw new Error('Unable to resolve ingestion host');
  }
};

/**
 * Check if an address is disallowed (private/internal).
 * @param {Object} addressInfo - Address info with address and family
 * @returns {boolean} True if disallowed
 */
export const isAddressDisallowed = ({
  address,
  family,
}: {
  address: string;
  family: number;
}): boolean => {
  if (family === 4) {
    return isPrivateIpv4(address);
  }
  if (family === 6) {
    return isPrivateIpv6(address);
  }
  return true;
};

interface ValidationError extends Error {
  statusCode?: number;
}

/**
 * Validate an ingestion URL for SSRF protection.
 * @param {string} rawUrl - URL to validate
 * @returns {Promise<string>} Validated and normalized URL
 * @throws {Error} If URL is invalid or targets private addresses
 */
export const validateIngestionUrl = async (rawUrl: string): Promise<string> => {
  if (!rawUrl || typeof rawUrl !== 'string') {
    const error: ValidationError = new Error('Ingestion URL is required');
    error.statusCode = 400;
    throw error;
  }

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch (error) {
    const err: ValidationError = new Error('Invalid ingestion URL format');
    err.statusCode = 400;
    throw err;
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    const error: ValidationError = new Error('Only HTTP and HTTPS ingestion URLs are allowed');
    error.statusCode = 400;
    throw error;
  }

  const hostname = parsed.hostname.toLowerCase();
  const disallowedHostnames = new Set(['localhost', '127.0.0.1', '::1']);
  if (disallowedHostnames.has(hostname)) {
    const error: ValidationError = new Error('Ingestion URL may not target local addresses');
    error.statusCode = 400;
    throw error;
  }

  const ipType = net.isIP(hostname);
  let addresses: Array<{ address: string; family: number }>;
  if (ipType) {
    const family = ipType === 6 ? 6 : 4;
    addresses = [{ address: hostname, family }];
  } else {
    addresses = await resolveHostAddresses(hostname);
  }

  if (!Array.isArray(addresses) || addresses.length === 0) {
    const error: ValidationError = new Error('Unable to resolve ingestion URL host');
    error.statusCode = 400;
    throw error;
  }

  if (addresses.some(isAddressDisallowed)) {
    const error: ValidationError = new Error(
      'Ingestion URL resolves to a private or disallowed address',
    );
    error.statusCode = 400;
    throw error;
  }

  return parsed.toString();
};
