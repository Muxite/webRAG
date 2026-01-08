const LOCALHOST_API = 'http://localhost:8080';
const AWS_API = 'https://euglena-api.com';
const API_MODE_KEY = 'euglena_api_mode';

type ApiMode = 'localhost' | 'aws' | 'auto';

function getApiBaseURL(): string {
  // Environment variable takes highest priority
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // Check for stored preference in localStorage
  if (typeof window !== 'undefined') {
    const storedMode = localStorage.getItem(API_MODE_KEY) as ApiMode | null;
    
    if (storedMode === 'localhost') {
      return LOCALHOST_API;
    }
    if (storedMode === 'aws') {
      return AWS_API;
    }
    // If 'auto' or null, use hostname detection
    if (storedMode === 'auto' || storedMode === null) {
      if (window.location.hostname === 'localhost' || 
          window.location.hostname === '127.0.0.1' ||
          window.location.hostname.startsWith('192.168.') ||
          window.location.hostname.startsWith('10.')) {
        return LOCALHOST_API;
      }
    }
  }
  
  return AWS_API;
}

export function setApiMode(mode: ApiMode): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem(API_MODE_KEY, mode);
    // Reload to apply the change
    window.location.reload();
  }
}

export function getApiMode(): ApiMode {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem(API_MODE_KEY) as ApiMode | null;
    return stored || 'auto';
  }
  return 'auto';
}

export function getCurrentApiBaseURL(): string {
  return getApiBaseURL();
}

export const API_CONFIG = {
  baseURL: getApiBaseURL(),
  LOCALHOST_API,
  AWS_API,
} as const;

