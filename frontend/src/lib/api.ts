/**
 * SARTHI Vision - API Client
 * Handles communication with the FastAPI backend.
 */

import type { ApiResponse, AnalysisResponse, ErrorResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Check if the backend is running and healthy.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}

/**
 * Upload a file for analysis. Returns the structured detection payload.
 *
 * @param file - The image or video file to analyze
 * @param confidence - Detection confidence threshold (0.1 to 1.0)
 * @param onProgress - Optional callback for upload progress (0-100)
 */
export async function analyzeMedia(
  file: File,
  confidence: number = 0.4,
  onProgress?: (percent: number) => void
): Promise<ApiResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE}/api/analyze?confidence=${confidence}`;

  // Use XMLHttpRequest for upload progress tracking
  if (onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          onProgress(percent);
        }
      });

      xhr.addEventListener("load", () => {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data as ApiResponse);
        } catch {
          reject(new Error("Failed to parse server response"));
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Network error: could not reach the analysis server"));
      });

      xhr.addEventListener("timeout", () => {
        reject(new Error("Request timed out. The file may be too large or the server is busy."));
      });

      xhr.open("POST", url);
      xhr.timeout = 600000; // 10 minute timeout for large videos
      xhr.send(formData);
    });
  }

  // Simple fetch for no-progress case
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  return data as ApiResponse;
}

/**
 * Type guard: check if the response is a successful analysis.
 */
export function isSuccess(response: ApiResponse): response is AnalysisResponse {
  return response.success === true;
}

/**
 * Type guard: check if the response is an error.
 */
export function isError(response: ApiResponse): response is ErrorResponse {
  return response.success === false;
}
