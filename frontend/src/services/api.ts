import { API_CONFIG } from '../config/api';
import { supabase } from './supabaseClient';

export interface TaskRequest {
  mandate: string;
  max_ticks?: number;
  correlation_id?: string;
}

export interface TaskResponse {
  correlation_id: string;
  status: string;
  mandate: string;
  created_at: string;
  updated_at: string;
  result?: {
    success?: boolean;
    deliverables?: string[];
    notes?: string;
    [key: string]: unknown;
  };
  error?: string;
  tick?: number;
  max_ticks: number;
}

class ApiClient {
  private baseURL: string;

  constructor() {
    this.baseURL = API_CONFIG.baseURL;
  }

  // Update base URL dynamically (for when API mode changes)
  updateBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
  }

  getBaseURL(): string {
    return this.baseURL;
  }

  private async withAuthHeaders(headers: HeadersInit = {}): Promise<HeadersInit> {
    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    const baseHeaders: HeadersInit = {
      'Content-Type': 'application/json',
      ...headers,
    };
    if (!token) {
      return baseHeaders;
    }
    return {
      ...baseHeaders,
      Authorization: `Bearer ${token}`,
    };
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const headers = await this.withAuthHeaders(options.headers || {});

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new Error(error.detail || 'Request failed');
      }

      return response.json();
    } catch (error) {
      // Handle network errors (e.g., failed to fetch on mobile devices or localhost server not running)
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        const isLocalhost = this.baseURL.includes('localhost') || this.baseURL.includes('127.0.0.1');
        if (isLocalhost) {
          throw new Error(`Cannot connect to API server at ${this.baseURL}. Make sure the gateway service is running (e.g., docker compose up gateway)`);
        } else {
          throw new Error(`Network error: Cannot reach ${this.baseURL}. Please check your internet connection.`);
        }
      }
      throw error;
    }
  }

  async submitTask(request: TaskRequest): Promise<TaskResponse> {
    return this.request<TaskResponse>('/tasks', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getTask(correlationId: string): Promise<TaskResponse> {
    return this.request<TaskResponse>(`/tasks/${correlationId}`);
  }
}

export const apiClient = new ApiClient();

