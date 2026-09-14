const TOKEN_KEY = "smartroute.jwt";
const TOKEN_EXPIRES_KEY = "smartroute.jwt.expiresAt";
const DEMO_SESSION_KEY = "smartroute.demoSessionId";

interface DemoTokenResponse {
  access_token: string;
  expires_at: number;
  session_id: string;
  token_type: string;
}

const getDemoSessionId = () => {
  const existing = localStorage.getItem(DEMO_SESSION_KEY);
  if (existing) return existing;

  const sessionId = crypto.randomUUID();
  localStorage.setItem(DEMO_SESSION_KEY, sessionId);
  return sessionId;
};

export const clearAuthToken = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_EXPIRES_KEY);
};

export const getAuthToken = () => {
  const token = localStorage.getItem(TOKEN_KEY)?.trim();
  if (!token) return null;

  const expiresAt = Number(localStorage.getItem(TOKEN_EXPIRES_KEY) || "0");
  if (expiresAt && Date.now() >= expiresAt * 1000) {
    clearAuthToken();
    return null;
  }

  return token;
};

export const ensureDemoAuthToken = async () => {
  const existing = getAuthToken();
  if (existing) return existing;

  const res = await fetch("/v1/auth/demo-token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: getDemoSessionId() }),
  });

  if (!res.ok) {
    throw new Error(`Demo authentication failed (HTTP ${res.status})`);
  }

  const data = (await res.json()) as DemoTokenResponse;
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(TOKEN_EXPIRES_KEY, String(data.expires_at));
  localStorage.setItem(DEMO_SESSION_KEY, data.session_id);
  return data.access_token;
};
