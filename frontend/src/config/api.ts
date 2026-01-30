const DEFAULT_LOCALHOST_PORT = '8080';
const LOCALHOST_API = `http://localhost:${import.meta.env.VITE_API_PORT || DEFAULT_LOCALHOST_PORT}`;
const AWS_API = import.meta.env.VITE_AWS_API_URL || 'https://euglena-api.com';
const API_MODE_KEY = 'euglena_api_mode';

type ApiMode = 'localhost' | 'aws' | 'auto';

function getApiBaseURL(): string {
  const envApiUrl = import.meta.env.VITE_API_BASE_URL;
  
  if (envApiUrl) {
    return envApiUrl;
  }
  
  if (typeof window !== 'undefined') {
    const storedMode = localStorage.getItem(API_MODE_KEY) as ApiMode | null;
    
    if (storedMode === 'localhost') {
      const customPort = import.meta.env.VITE_API_PORT || DEFAULT_LOCALHOST_PORT;
      return `http://localhost:${customPort}`;
    }
    if (storedMode === 'aws') {
      return AWS_API;
    }
    if (storedMode === 'auto' || storedMode === null) {
      if (window.location.hostname === 'localhost' || 
          window.location.hostname === '127.0.0.1' ||
          window.location.hostname.startsWith('192.168.') ||
          window.location.hostname.startsWith('10.')) {
        const customPort = import.meta.env.VITE_API_PORT || DEFAULT_LOCALHOST_PORT;
        return `http://localhost:${customPort}`;
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

