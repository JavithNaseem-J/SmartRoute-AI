export const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export const formatUploadedAt = (value?: string | null) => {
  if (!value) return "Just now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Recently";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

const formatSourceLabel = (source: string) => {
  const withoutPrefix = source.replace(/^Source\s+\d+:\s*/i, "").trim();
  const pageMatch = withoutPrefix.match(/\s+—\s+page\s+\d+$/i);
  const pageSuffix = pageMatch?.[0] ?? "";
  const basePath = pageSuffix ? withoutPrefix.slice(0, -pageSuffix.length) : withoutPrefix;
  const filename = basePath.split("/").pop() || basePath;
  const readable = filename.replace(
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-/i,
    ""
  );

  try {
    return `${decodeURIComponent(readable)}${pageSuffix}`;
  } catch {
    return `${readable}${pageSuffix}`;
  }
};

export const uniqueSources = (sources: string[] = []) =>
  Array.from(new Set(sources.map(formatSourceLabel))).filter(Boolean);
