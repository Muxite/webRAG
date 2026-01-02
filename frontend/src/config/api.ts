function getApiBaseURL(): string {

  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  if (typeof window !== 'undefined' && 
      (window.location.hostname === 'localhost' || 
       window.location.hostname === '127.0.0.1' ||
       window.location.hostname.startsWith('192.168.') ||
       window.location.hostname.startsWith('10.'))) {
    return 'http://localhost:8080';
  }
  
  return 'https://euglena-api.com';
}

export const API_CONFIG = {
  baseURL: getApiBaseURL(),
} as const;

