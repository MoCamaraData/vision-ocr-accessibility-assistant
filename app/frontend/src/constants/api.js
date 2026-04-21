const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_BASE  = API_BASE.replace(/^http/, "ws");

export const API = {
  health:      `${API_BASE}/health`,
  ocrImage:    `${API_BASE}/ocr/image`,
  ocrStream:   `${WS_BASE}/ocr/stream`,
};
