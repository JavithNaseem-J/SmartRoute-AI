import { ensureDemoAuthToken, getAuthToken } from "@/lib/auth";

export interface StoredDocument {
  id: string;
  filename: string;
  content_type?: string | null;
  size_bytes: number;
  storage_bucket: string;
  storage_path: string;
  created_at?: string | null;
  modified_time?: number | null;
}

const authHeaders = async () => {
  const token = getAuthToken() || (await ensureDemoAuthToken());
  return { Authorization: `Bearer ${token}` };
};

const readError = async (res: Response, fallback: string) => {
  const body = (await res.json().catch(() => ({}))) as { detail?: string };
  return body.detail || fallback;
};

export const listDocuments = async () => {
  const res = await fetch("/v1/documents", { headers: await authHeaders() });
  if (!res.ok) {
    throw new Error(await readError(res, `Document list failed (HTTP ${res.status})`));
  }
  return (await res.json()) as { documents: StoredDocument[]; total: number };
};

export const deleteDocument = async (filename: string) => {
  const res = await fetch(`/v1/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(await readError(res, `Document delete failed (HTTP ${res.status})`));
  }
  return res.json();
};

export const clearDocuments = async () => {
  const res = await fetch("/v1/documents", {
    method: "DELETE",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(await readError(res, `Document clear failed (HTTP ${res.status})`));
  }
  return res.json();
};
