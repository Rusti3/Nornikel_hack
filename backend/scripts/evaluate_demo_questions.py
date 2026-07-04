from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "backend").is_dir() else BACKEND_ROOT
DEFAULT_FIXTURE = BACKEND_ROOT / "pilot" / "demo_questions.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_local_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def request_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, timeout: float = 60) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def api_keys() -> tuple[str, ...]:
    values = []
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"):
        raw = os.getenv(name, "")
        for item in raw.replace(";", ",").replace("\n", ",").split(","):
            value = item.strip()
            if value and value not in values and not value.startswith("<"):
                values.append(value)
    return tuple(values)


def openrouter_preflight() -> list[dict[str, Any]]:
    results = []
    for slot, key in enumerate(api_keys()):
        try:
            value = request_json(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            ).get("data") or {}
            results.append({
                "slot": slot,
                "ok": True,
                "is_free_tier": value.get("is_free_tier"),
                "limit": value.get("limit"),
                "limit_remaining": value.get("limit_remaining"),
                "usage": value.get("usage"),
            })
        except Exception as exc:
            results.append({"slot": slot, "ok": False, "error_type": type(exc).__name__})
    return results


def openrouter_model_available(model: str) -> bool:
    try:
        rows = request_json("https://openrouter.ai/api/v1/models", timeout=30).get("data") or []
        return any(item.get("id") == model for item in rows)
    except Exception:
        return False


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = re.sub(r"\s*%", "%", value)
    return re.sub(r"\s+", " ", value)


def fact_matches(fact: str, answer: str) -> bool:
    wanted = re.findall(r"[a-zа-я0-9]+", normalize(fact))
    actual = re.findall(r"[a-zа-я0-9]+", answer)
    if not wanted:
        return False
    matched = 0
    for token in wanted:
        if token.isdigit() or len(token) <= 3:
            ok = token in actual
        else:
            prefix = token[:5] if re.search(r"[а-я]", token) else token[:4]
            ok = any(value.startswith(prefix) for value in actual)
        matched += int(ok)
    return matched / len(wanted) >= 0.7


def auto_review(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result") or {}
    answer = normalize(str(result.get("answer_markdown") or ""))
    sources = result.get("sources") or []
    document = question["_document"]
    expected_file = normalize(document["file_name"])
    source_recall = any(
        expected_file in normalize(str(item.get("file_name") or item.get("title") or ""))
        or item.get("document_id") == document["document_id"]
        for item in sources
    )
    fact_checks = []
    for fact in question["required_facts"]:
        fact_checks.append({"fact": fact, "matched": fact_matches(fact, answer)})
    fact_recall = sum(item["matched"] for item in fact_checks) / max(1, len(fact_checks))
    citations = set(re.findall(r"\[S\d+\]", str(result.get("answer_markdown") or "")))
    mode_ok = result.get("mode") in question.get("acceptable_modes", [])
    terminal = response.get("status") in {"complete", "failed", "cancelled"}
    warnings = result.get("warnings") or []
    llm_ok = not any("Agent LLM" in str(warning) for warning in warnings)
    passed = bool(terminal and response.get("status") == "complete" and source_recall
                  and fact_recall >= 0.8 and citations and mode_ok and llm_ok)
    return {
        "pass": passed,
        "terminal": terminal,
        "source_recall": source_recall,
        "fact_recall": round(fact_recall, 3),
        "citation_count": len(citations),
        "mode_ok": mode_ok,
        "llm_ok": llm_ok,
        "fact_checks": fact_checks,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed real Agentic RAG demo evaluation")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "demo-eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--job-timeout", type=float, default=1800)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    load_local_env(ROOT / ".env")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in fixture["documents"]}
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        preflight = openrouter_preflight() if args.preflight else []
        checkpoint = {
            "run_id": args.run_id,
            "created_at": utc_now(),
            "provider": os.getenv("AGENT_LLM_PROVIDER", "yandex"),
            "model": os.getenv("OPENROUTER_MODEL") if os.getenv("AGENT_LLM_PROVIDER") == "openrouter" else os.getenv("LLM_MODEL"),
            "preflight": preflight,
            "model_available": (
                openrouter_model_available(os.getenv("OPENROUTER_MODEL", ""))
                if args.preflight and os.getenv("AGENT_LLM_PROVIDER") == "openrouter" else None
            ),
            "results": {},
        }
        atomic_json(checkpoint_path, checkpoint)

    if args.preflight and checkpoint.get("provider") == "openrouter":
        checks = checkpoint.get("preflight") or []
        if not checks or not any(item.get("ok") for item in checks):
            checkpoint["blocked_reason"] = "No valid local OpenRouter key slots; add fresh keys to ignored .env"
            checkpoint["updated_at"] = utc_now()
            atomic_json(checkpoint_path, checkpoint)
            print(json.dumps({"status": "blocked", "reason": checkpoint["blocked_reason"]}, ensure_ascii=False))
            return
        if checkpoint.get("model_available") is False:
            checkpoint["blocked_reason"] = "Configured OpenRouter model is not currently listed"
            checkpoint["updated_at"] = utc_now()
            atomic_json(checkpoint_path, checkpoint)
            print(json.dumps({"status": "blocked", "reason": checkpoint["blocked_reason"]}, ensure_ascii=False))
            return
        remaining = [item.get("limit_remaining") for item in checks if item.get("ok")]
        numeric_remaining = [float(item) for item in remaining if isinstance(item, (int, float))]
        if numeric_remaining and max(numeric_remaining) <= 0:
            checkpoint["blocked_reason"] = "All OpenRouter key slots report exhausted quota"
            checkpoint["updated_at"] = utc_now()
            atomic_json(checkpoint_path, checkpoint)
            print(json.dumps({"status": "blocked", "reason": checkpoint["blocked_reason"]}, ensure_ascii=False))
            return

    questions = fixture["questions"][: args.limit or None]
    endpoint = args.base_url.rstrip("/") + "/api/mekg/v1/agentic_rag/jobs"
    for index, raw_question in enumerate(questions, start=1):
        if raw_question["id"] in checkpoint["results"]:
            continue
        question = {**raw_question, "_document": documents[raw_question["document_id"]]}
        started = time.monotonic()
        accepted = request_json(endpoint, method="POST", payload={
            "query": question["query"],
            "allow_external_web": False,
            "corpora": [question["_document"]["corpus_id"]],
            "max_iterations": 3,
            "answer_language": "ru",
            "include_debug": True,
        })
        job_id = accepted["job_id"]
        status_url = endpoint + "/" + job_id
        while True:
            response = request_json(status_url, timeout=60)
            if response.get("status") in {"complete", "failed", "cancelled"}:
                break
            if time.monotonic() - started > args.job_timeout:
                response = {"job_id": job_id, "status": "timeout", "result": None, "error": "runner timeout"}
                break
            time.sleep(args.poll_seconds)
        sanitized = {
            "question_id": question["id"],
            "job_id": job_id,
            "started_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "response": response,
        }
        sanitized["auto_review"] = auto_review(question, response)
        sanitized["manual_verdict"] = "NOT_REVIEWED"
        checkpoint["results"][question["id"]] = sanitized
        checkpoint["updated_at"] = utc_now()
        atomic_json(checkpoint_path, checkpoint)
        print(json.dumps({
            "progress": f"{index}/{len(questions)}", "question_id": question["id"],
            "status": response.get("status"), "auto_pass": sanitized["auto_review"]["pass"],
        }, ensure_ascii=False), flush=True)
        warnings = ((response.get("result") or {}).get("warnings") or [])
        if checkpoint.get("provider") == "openrouter" and any(
            "Agent LLM" in str(warning) for warning in warnings
        ):
            checkpoint["blocked_reason"] = "OpenRouter became unavailable; resume after quota/provider recovery"
            checkpoint["updated_at"] = utc_now()
            atomic_json(checkpoint_path, checkpoint)
            break

    checks = [item.get("auto_review") or {} for item in checkpoint["results"].values()]
    summary = {
        "run_id": args.run_id,
        "questions_total": len(questions),
        "questions_finished": len(checkpoint["results"]),
        "auto_passed": sum(bool(item.get("pass")) for item in checks),
        "source_recall": round(sum(bool(item.get("source_recall")) for item in checks) / max(1, len(checks)), 3),
        "critical_hallucinations": None,
        "manual_review_complete": False,
        "checkpoint": str(checkpoint_path),
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
