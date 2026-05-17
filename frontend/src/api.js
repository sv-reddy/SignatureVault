const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

export async function startAnalysis({ requestId, vaultFiles, questionedFile }) {
  const formData = new FormData();
  formData.append("requestId", requestId);

  vaultFiles.forEach((file) => {
    formData.append("vault", file);
  });

  formData.append("questioned", questionedFile);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Analysis request failed");
  }
  return data;
}

export async function fetchProgress(jobId) {
  const response = await fetch(`${API_BASE}/api/progress/${jobId}`);
  if (!response.ok) {
    throw new Error("Progress unavailable");
  }
  return response.json();
}

export function resolveAsset(pathOrUrl) {
  if (!pathOrUrl) {
    return "";
  }
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  if (pathOrUrl.startsWith("/")) {
    return `${API_BASE}${pathOrUrl}`;
  }
  return pathOrUrl;
}
