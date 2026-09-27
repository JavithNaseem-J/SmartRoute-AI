import type { Citation } from "@/types/chat";

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
  const locationMatch = withoutPrefix.match(/\s+(?:—|-)\s+(?:page\s+\d+|section\s+.+)$/i);
  const locationSuffix = locationMatch?.[0] ?? "";
  const basePath = locationSuffix
    ? withoutPrefix.slice(0, -locationSuffix.length)
    : withoutPrefix;
  const filename = basePath.split("/").pop() || basePath;
  const readable = filename.replace(
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-/i,
    ""
  );

  try {
    return `${decodeURIComponent(readable)}${locationSuffix}`;
  } catch {
    return `${readable}${locationSuffix}`;
  }
};

export const uniqueSources = (sources: string[] = []) =>
  Array.from(new Set(sources.map(formatSourceLabel))).filter(Boolean);

export const legacySourcesToCitations = (sources: string[] = []): Citation[] => {
  const citations = uniqueSources(sources).map((source, index) => {
    const pageMatch = source.match(/\s+(?:—|-)\s+page\s+(\d+)$/i);
    const sectionMatch = source.match(/\s+(?:—|-)\s+section\s+(.+)$/i);
    const suffix = pageMatch?.[0] ?? sectionMatch?.[0] ?? "";
    return {
      id: `C${index + 1}`,
      filename: suffix ? source.slice(0, -suffix.length) : source,
      page: pageMatch ? Number(pageMatch[1]) : null,
      section: sectionMatch?.[1] ?? null,
      excerpt: "",
    };
  });

  return citations.filter(
    (citation, index) =>
      citations.findIndex(
        (candidate) =>
          candidate.filename === citation.filename &&
          candidate.page === citation.page &&
          candidate.section === citation.section
      ) === index
  );
};
