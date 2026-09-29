"""Exercise upload, retrieval, generation, and citations with synthetic documents."""

import argparse
import io
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, TextStringObject

DATASET = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "rag_live_eval.json"
MAX_LIVE_QUERIES = 15
ABSTENTION = re.compile(
    r"(?:do(?:es)? not contain|not (?:provided|mentioned|available|found)|"
    r"no (?:information|phone number)|cannot determine|can't find|couldn't find)",
    re.IGNORECASE,
)


def pdf_bytes(pages: list[str]) -> bytes:
    """Build a small text PDF so the benchmark exercises real page extraction."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        literal = io.BytesIO()
        TextStringObject(text).write_to_stream(literal)
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 720 Td " + literal.getvalue() + b" Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def upload_files(documents: list[dict]) -> list[tuple[str, tuple[str, bytes, str]]]:
    files = []
    for document in documents:
        filename = document["filename"]
        if filename.endswith(".pdf"):
            content = pdf_bytes(document["pages"])
            content_type = "application/pdf"
        elif filename.endswith((".md", ".txt")):
            content = document["content"].encode("utf-8")
            content_type = "text/markdown" if filename.endswith(".md") else "text/plain"
        else:
            raise ValueError(f"Unsupported benchmark file: {filename}")
        files.append(("files", (filename, content, content_type)))
    return files


def score_case(case: dict, response: dict) -> dict:
    answer = str(response.get("answer") or "")
    normalized_answer = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", answer))
    citations = response.get("citations") or []
    checks = {"request_succeeded": response.get("success") is True}

    if case.get("answerable", True):
        filename = case["filename"]
        page = case["page"]
        location = f"{filename} - page {page}" if page is not None else filename
        checks["retrieved_gold_source"] = any(
            location in str(source) for source in response.get("sources") or []
        )
        checks["expected_facts"] = all(
            re.search(pattern, normalized_answer, re.IGNORECASE) is not None
            for pattern in case["answer_patterns"]
        )
        checks["grounded_citations"] = bool(citations) and all(
            citation.get("filename") == filename
            and citation.get("page") == page
            and case["evidence"].lower() in str(citation.get("excerpt") or "").lower()
            and f"[{citation.get('id')}]" in answer
            for citation in citations
        )
    else:
        checks["abstained"] = bool(ABSTENTION.search(normalized_answer))
        checks["no_citations"] = not citations

    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "answer": answer,
        "citations": citations,
        "sources": response.get("sources") or [],
        "cost": response.get("cost") or 0,
    }


def run_benchmark(client: httpx.Client, dataset: dict) -> dict:
    cases = dataset["cases"]
    if not 1 <= len(cases) <= MAX_LIVE_QUERIES:
        raise ValueError(f"Live benchmark requires 1-{MAX_LIVE_QUERIES} questions.")

    auth = client.post("/v1/auth/demo-token", json={"session_id": str(uuid4())})
    auth.raise_for_status()
    headers = {"Authorization": f"Bearer {auth.json()['access_token']}"}
    existing = client.get("/v1/documents", headers=headers)
    existing.raise_for_status()
    if existing.json()["total"] != 0:
        raise RuntimeError("The evaluation session unexpectedly contains documents.")

    uploaded = False
    results: list[dict] = []
    cleanup = "not_needed"
    try:
        uploaded = True
        upload = client.post(
            "/v1/documents/upload", headers=headers, files=upload_files(dataset["documents"])
        )
        upload.raise_for_status()
        for case in cases:
            try:
                reply = client.post(
                    "/v1/query",
                    headers=headers,
                    json={
                        "query": case["question"],
                        "strategy": "quality_first",
                        "use_retrieval": True,
                    },
                )
                response = (
                    reply.json()
                    if reply.status_code == 200
                    else {
                        "success": False,
                        "error": f"HTTP {reply.status_code}",
                    }
                )
            except (httpx.HTTPError, ValueError) as exc:
                response = {"success": False, "error": type(exc).__name__}
            results.append(score_case(case, response))
    finally:
        if uploaded:
            try:
                deletion = client.delete("/v1/documents", headers=headers)
                deletion.raise_for_status()
                remaining = client.get("/v1/documents", headers=headers)
                remaining.raise_for_status()
                cleanup = "ok" if remaining.json()["total"] == 0 else "failed"
            except (httpx.HTTPError, ValueError, KeyError):
                cleanup = "failed"

    passed = sum(result["passed"] for result in results)
    return {
        "benchmark": "live_end_to_end_synthetic",
        "cases": len(cases),
        "passed": passed,
        "pass_rate": passed / len(cases),
        "total_reported_cost_usd": round(sum(float(item["cost"]) for item in results), 6),
        "cleanup": cleanup,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Running SmartRoute API URL")
    parser.add_argument(
        "--confirm-live", action="store_true", help="Allow live uploads and LLM calls"
    )
    parser.add_argument("--case-ids", help="Comma-separated case IDs for a bounded follow-up run")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("Pass --confirm-live to allow up to 15 live LLM requests.")
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        parser.error("Use HTTPS for a remote evaluation target.")

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.case_ids:
        selected = set(args.case_ids.split(","))
        available = {case["id"] for case in dataset["cases"]}
        if not selected <= available:
            parser.error(f"Unknown case IDs: {', '.join(sorted(selected - available))}")
        dataset["cases"] = [case for case in dataset["cases"] if case["id"] in selected]
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120) as client:
        report = run_benchmark(client, dataset)
    if args.case_ids:
        report["benchmark"] = "live_end_to_end_synthetic_subset"
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["cases"] and report["cleanup"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
