from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT.parent
DEFAULT_FIXTURE = BACKEND_ROOT / "pilot" / "expert_questions.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 90) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    for attempt in range(4):
        request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt >= 3:
                raise
            time.sleep(min(5.0, 0.5 * (2 ** attempt)))
    raise RuntimeError("unreachable HTTP retry state")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е"))


def review(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result") or {}
    answer = normalize(str(result.get("answer_markdown") or ""))
    groups = question.get("required_term_groups") or []
    checks = [
        {"terms": group, "matched": any(normalize(term) in answer for term in group)}
        for group in groups
    ]
    citations = len(set(re.findall(r"\[S\d+\]", str(result.get("answer_markdown") or ""))))
    sources = result.get("sources") or []
    forbidden = [item for item in question.get("forbidden_claims") or [] if normalize(item) in answer]
    recall = sum(bool(item["matched"]) for item in checks) / max(1, len(checks))
    acceptable = result.get("mode") in question.get("acceptable_modes", [])
    passed = response.get("status") == "complete" and acceptable and bool(sources) and citations > 0 and recall >= 0.7 and not forbidden
    return {
        "pass": passed,
        "term_recall": round(recall, 3),
        "citation_count": citations,
        "source_count": len(sources),
        "mode_ok": acceptable,
        "forbidden_matches": forbidden,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed expert-question Agentic RAG evaluation")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--run-id", default=datetime.now().strftime("expert-%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "expert-eval")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated question IDs for a targeted regression run")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    questions = json.loads(args.fixture.read_text(encoding="utf-8"))["questions"]
    selected_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
    if selected_ids:
        questions = [item for item in questions if item["id"] in selected_ids]
    questions = questions[:args.limit or None]
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else {
        "run_id": args.run_id, "created_at": now(), "web": args.web, "results": {},
    }
    endpoint = args.base_url.rstrip("/") + "/api/mekg/v1/agentic_rag/jobs"
    for index, question in enumerate(questions, start=1):
        if question["id"] in checkpoint["results"]:
            continue
        started = time.monotonic()
        accepted = request_json(endpoint, method="POST", payload={
            "query": question["query"],
            "allow_external_web": args.web,
            "web_profile_ids": ["journals", "mining_metals"] if args.web else [],
            "max_iterations": 3,
            "answer_language": "ru",
            "include_debug": True,
        })
        job_id = accepted["job_id"]
        while True:
            response = request_json(f"{endpoint}/{job_id}")
            if response.get("status") in {"complete", "failed", "cancelled"}:
                break
            if time.monotonic() - started > args.timeout:
                response = {"job_id": job_id, "status": "timeout", "error": "evaluation timeout"}
                break
            time.sleep(2)
        item = {
            "question_id": question["id"], "job_id": job_id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "response": response,
            "auto_review": review(question, response),
            "manual_verdict": "NOT_REVIEWED",
            "manual_notes": "",
        }
        checkpoint["results"][question["id"]] = item
        checkpoint["updated_at"] = now()
        atomic_json(checkpoint_path, checkpoint)
        print(json.dumps({"progress": f"{index}/{len(questions)}", "id": question["id"], "pass": item["auto_review"]["pass"]}, ensure_ascii=False), flush=True)
    checks = [item["auto_review"] for item in checkpoint["results"].values()]
    summary = {
        "run_id": args.run_id, "web": args.web,
        "finished": len(checks), "total": len(questions),
        "auto_passed": sum(bool(item.get("pass")) for item in checks),
        "critical_hallucinations": sum(bool(item.get("forbidden_matches")) for item in checks),
        "manual_review_complete": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
