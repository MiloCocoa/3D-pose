const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 25000;

/**
 * Call the FastAPI /analyze endpoint.
 * @param {object} payload - JSON payload that matches PoseData schema.
 * @returns {Promise<object>} Parsed JSON response.
 */
export async function analyzePose(payload) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await safeParseError(response);
      throw new Error(detail ?? `Request failed with status ${response.status}`);
    }

    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

async function safeParseError(response) {
  try {
    const data = await response.json();
    return data?.detail || JSON.stringify(data);
  } catch {
    return null;
  }
}

export function getApiBaseUrl() {
  return API_BASE_URL;
}

