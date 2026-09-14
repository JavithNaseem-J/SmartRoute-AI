import mimetypes
import os
import re
from dataclasses import dataclass
from typing import Dict
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import aiohttp


_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_segment(value: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("-", value.strip()).strip(".-")
    return cleaned or "document"


def guess_content_type(filename: str, fallback: str = "application/octet-stream") -> str:
    return mimetypes.guess_type(filename)[0] or fallback


def normalize_supabase_url(value: str) -> str:
    """Return the project origin even if a REST/Storage endpoint was pasted."""
    raw_url = value.strip().rstrip("/")
    parsed = urlsplit(raw_url)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return raw_url


@dataclass(frozen=True)
class SupabaseStorage:
    url: str
    service_role_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> "SupabaseStorage":
        missing = [
            name
            for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_STORAGE_BUCKET")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                f"Missing Supabase Storage environment variables: {', '.join(missing)}"
            )
        return cls(
            url=normalize_supabase_url(os.environ["SUPABASE_URL"]),
            service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip(),
            bucket=os.environ["SUPABASE_STORAGE_BUCKET"].strip(),
        )

    @property
    def _storage_url(self) -> str:
        return f"{self.url}/storage/v1"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
        }

    def object_path(self, user_id: str, filename: str) -> str:
        return f"{safe_segment(user_id)}/{uuid4()}-{safe_segment(filename)}"

    def _object_url(self, path: str) -> str:
        encoded_path = quote(path, safe="/")
        return f"{self._storage_url}/object/{self.bucket}/{encoded_path}"

    async def upload(self, path: str, content: bytes, content_type: str) -> None:
        headers = {
            **self._headers,
            "Content-Type": content_type,
            "cache-control": "3600",
            "x-upsert": "false",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._object_url(path), data=content, headers=headers
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(
                        f"Supabase Storage upload failed: {response.status} {detail}"
                    )

    async def download(self, path: str) -> bytes:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(self._object_url(path)) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(
                        f"Supabase Storage download failed: {response.status} {detail}"
                    )
                return await response.read()

    async def delete(self, path: str) -> None:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.delete(self._object_url(path)) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(
                        f"Supabase Storage delete failed: {response.status} {detail}"
                    )
