import io
import json
from pathlib import Path

import httpx
import pytest
from pypdf import PdfReader

from scripts.run_live_rag_eval import DATASET, pdf_bytes, run_benchmark, score_case


def test_pdf_fixture_preserves_page_text():
    pdf = PdfReader(io.BytesIO(pdf_bytes(["Page one answer.", "Page two evidence."])))
    assert [page.extract_text().strip() for page in pdf.pages] == [
        "Page one answer.",
        "Page two evidence.",
    ]


def test_live_dataset_has_a_bounded_number_of_cases():
    dataset = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    assert len(dataset["documents"]) >= 2
    assert 1 <= len(dataset["cases"]) <= 15
    assert len({case["id"] for case in dataset["cases"]}) == len(dataset["cases"])
    assert any(case.get("answerable") is False for case in dataset["cases"])


def test_score_case_requires_correct_page_and_evidence():
    case = {
        "id": "role",
        "answer_patterns": ["backend engineer"],
        "filename": "Cover.pdf",
        "page": 2,
        "evidence": "Backend Engineer position",
    }
    response = {
        "success": True,
        "answer": "Jordan seeks the Backend Engineer role. [C1]",
        "sources": ["Source 1: Cover.pdf - page 2"],
        "citations": [
            {
                "id": "C1",
                "filename": "Cover.pdf",
                "page": 2,
                "excerpt": "Applying for the Backend Engineer position.",
            }
        ],
    }
    assert score_case(case, response)["passed"] is True

    response["citations"][0]["page"] = 1
    assert score_case(case, response)["checks"]["grounded_citations"] is False
    response["citations"][0]["page"] = 2
    response["answer"] = "Jordan seeks the Backend Engineer role."
    assert score_case(case, response)["checks"]["grounded_citations"] is False


def test_score_case_accepts_provider_narrow_spaces():
    case = {
        "id": "company",
        "answer_patterns": ["cedarline systems"],
        "filename": "Cover.pdf",
        "page": 1,
        "evidence": "Cedarline Systems",
    }
    response = {
        "success": True,
        "answer": "Jordan applies to Cedarline\u202fSystems [C1].",
        "sources": ["Source 1: Cover.pdf - page 1"],
        "citations": [
            {
                "id": "C1",
                "filename": "Cover.pdf",
                "page": 1,
                "excerpt": "Applying to Cedarline Systems.",
            }
        ],
    }
    assert score_case(case, response)["passed"] is True


def test_unanswerable_case_requires_abstention_without_citation():
    case = {"id": "missing", "answerable": False}
    response = {
        "success": True,
        "answer": "The uploaded documents do not contain a phone number.",
        "citations": [],
    }
    assert score_case(case, response)["passed"] is True
    response["citations"] = [{"id": "C1"}]
    assert score_case(case, response)["passed"] is False


@pytest.mark.parametrize("query_failed", [False, True])
def test_benchmark_uploads_queries_and_cleans_isolated_session(query_failed):
    state = {"documents": 0, "queries": 0, "deleted": False}
    dataset = {
        "documents": [{"filename": "Cover.pdf", "pages": ["Backend Engineer position."]}],
        "cases": [
            {
                "id": "role",
                "question": "Which role?",
                "answer_patterns": ["backend engineer"],
                "filename": "Cover.pdf",
                "page": 1,
                "evidence": "Backend Engineer position",
            },
            {"id": "unknown", "question": "What is the phone number?", "answerable": False},
        ],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/auth/demo-token":
            return httpx.Response(200, json={"access_token": "test-token"})
        assert request.headers["Authorization"] == "Bearer test-token"
        if path == "/v1/documents" and request.method == "GET":
            return httpx.Response(200, json={"total": state["documents"], "documents": []})
        if path == "/v1/documents/upload":
            assert b"Cover.pdf" in request.content
            state["documents"] = 1
            return httpx.Response(200, json={"documents": [{"filename": "Cover.pdf"}]})
        if path == "/v1/query":
            state["queries"] += 1
            if state["queries"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "answer": "The Backend Engineer position. [C1]",
                        "sources": ["Source 1: Cover.pdf - page 1"],
                        "citations": [
                            {
                                "id": "C1",
                                "filename": "Cover.pdf",
                                "page": 1,
                                "excerpt": "Backend Engineer position.",
                            }
                        ],
                    },
                )
            if query_failed:
                return httpx.Response(500)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "answer": "The documents do not contain a phone number.",
                    "citations": [],
                },
            )
        if path == "/v1/documents" and request.method == "DELETE":
            state["documents"] = 0
            state["deleted"] = True
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404)

    with httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(respond)
    ) as client:
        report = run_benchmark(client, dataset)

    assert report["passed"] == (1 if query_failed else 2)
    assert report["cleanup"] == "ok"
    assert state["queries"] == 2
    assert state["deleted"] is True


def test_benchmark_rejects_more_than_fifteen_live_questions():
    dataset = {"cases": [{"id": str(index)} for index in range(16)]}
    with httpx.Client(base_url="https://example.test") as client:
        try:
            run_benchmark(client, dataset)
        except ValueError as exc:
            assert "1-15" in str(exc)
        else:
            raise AssertionError("The live query limit was not enforced")
