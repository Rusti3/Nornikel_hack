from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
from typing import Any

from .agentic import AgentCancelled, AgenticRAG
from .config import MEKGConfig
from .models import AgenticRAGRequest
from .search_store import SearchStore
from .service import MEKGService


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mekg.agent_worker")


class AgentWorker:
    def __init__(self, config: MEKGConfig | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self.store = SearchStore(self.config)
        self.service = MEKGService(self.config)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        self.stop_event = threading.Event()

    def stop(self, *_args: Any) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        self.store.initialize_schema()
        logger.info("Agent worker started as %s", self.owner)
        while not self.stop_event.is_set():
            row = self.store.claim_agent_run(
                self.owner, run_types=["agentic_rag", "graph_qa"]
            )
            if not row:
                self.stop_event.wait(self.config.agent_poll_seconds)
                continue
            self._run_claimed(row)
        self.store.close()

    def _run_claimed(self, row: dict[str, Any]) -> None:
        run_id = row["run_id"]
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, args=(run_id, heartbeat_stop), daemon=True
        )
        heartbeat.start()
        try:
            if row["run_type"] == "graph_qa":
                self.store.append_agent_event(run_id, "qa_started", {"status": "running"})
                result = self.service.qa()
                state = {"qa": result}
                self.store.complete_agent_run(run_id, result, state)
                self.store.append_agent_event(
                    run_id, "completed", {"passed": result.get("passed"), "status": "complete"}
                )
                return
            request = AgenticRAGRequest.model_validate(row["request_json"])
            cache_value = json.dumps(
                {
                    "request": request.model_dump(mode="json"),
                    "corpus_version": self.store.corpus_version(),
                    "evidence_policy_version": 4,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            cached = self.store.cache_get("agent_answer", cache_value)
            if cached and cached.get("result") and cached.get("state"):
                result = json.loads(json.dumps(cached["result"], ensure_ascii=False))
                state = json.loads(json.dumps(cached["state"], ensure_ascii=False))
                result["job_id"] = run_id
                state["query_id"] = run_id
                self.store.append_agent_event(run_id, "cache_hit", {"status": "complete"})
                self.store.complete_agent_run(run_id, result, state)
                self.store.append_agent_event(
                    run_id, "completed", {
                        "mode": result.get("mode"), "confidence": result.get("confidence"),
                        "cached": True,
                    }
                )
                return
            orchestrator = AgenticRAG(
                self.config, store=self.store, service=self.service
            )
            result, state = orchestrator.run(
                run_id,
                request,
                owner=self.owner,
                initial_state=row.get("state_json") or None,
            )
            self.store.complete_agent_run(run_id, result, state.model_dump(mode="json"))
            self.store.cache_set(
                "agent_answer",
                cache_value,
                {"result": result, "state": state.model_dump(mode="json")},
                hours=24,
            )
        except AgentCancelled:
            self.store.fail_agent_run(run_id, "Cancelled by user", retryable=False)
            self.store.append_agent_event(run_id, "cancelled", {"status": "cancelled"})
        except Exception as exc:
            retryable = any(
                token in type(exc).__name__.casefold()
                for token in ("timeout", "connection", "serviceunavailable", "transient")
            )
            status = self.store.fail_agent_run(
                run_id, f"{type(exc).__name__}: {str(exc)[:800]}", retryable=retryable
            )
            self.store.append_agent_event(
                run_id,
                "retry_scheduled" if status == "queued" else "failed",
                {"status": status, "error_type": type(exc).__name__},
            )
            logger.exception("Agent run %s failed", run_id)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)

    def _heartbeat_loop(self, run_id: str, stop_event: threading.Event) -> None:
        interval = max(30.0, self.config.agent_lease_seconds / 3)
        while not stop_event.wait(interval):
            try:
                self.store.heartbeat_agent_run(run_id, self.owner)
            except Exception:
                logger.warning("Heartbeat failed for %s", run_id, exc_info=True)


def main() -> None:
    worker = AgentWorker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
