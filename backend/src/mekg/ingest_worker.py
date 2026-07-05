from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
from typing import Any

from .config import MEKGConfig
from .interactive_ingest import IngestCancelled, InteractiveIngestor
from .search_store import SearchStore


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mekg.ingest_worker")


class IngestWorker:
    def __init__(self, config: MEKGConfig | None = None) -> None:
        self.config = config or MEKGConfig.from_env()
        self.store = SearchStore(self.config)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        self.stop_event = threading.Event()

    def stop(self, *_args: Any) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        self.store.initialize_schema()
        logger.info("Ingest worker started as %s", self.owner)
        while not self.stop_event.is_set():
            row = self.store.claim_agent_run(
                self.owner, run_types=["document_ingest"]
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
            ingestor = InteractiveIngestor(self.config, store=self.store)
            ingestor.owner = self.owner
            result, state = asyncio.run(ingestor.run(
                run_id,
                row.get("request_json") or {},
                initial_state=row.get("state_json") or None,
            ))
            self.store.complete_agent_run(run_id, result, state)
        except IngestCancelled:
            self.store.fail_agent_run(run_id, "Cancelled by user", retryable=False)
            self.store.append_agent_event(run_id, "cancelled", {"status": "cancelled"})
        except Exception as exc:
            folded = f"{type(exc).__name__}: {exc}".casefold()
            retryable = any(token in folded for token in (
                "timeout", "connection", "serviceunavailable", "transient", "429", "temporarily",
            ))
            status = self.store.fail_agent_run(
                run_id, f"{type(exc).__name__}: {str(exc)[:800]}", retryable=retryable
            )
            self.store.append_agent_event(
                run_id,
                "retry_scheduled" if status == "queued" else "failed",
                {"status": status, "error_type": type(exc).__name__},
            )
            logger.exception("Ingest run %s failed", run_id)
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
    worker = IngestWorker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
