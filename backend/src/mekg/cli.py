from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .service import MEKGService
from .pipeline import FullCorpusPipeline, benchmark_fast_parser
from .search_store import SearchStore


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MEKG pilot runner")
    sub = result.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest-manifest")
    ingest.add_argument("manifest", nargs="?", default="/code/pilot/manifest.json")
    ingest.add_argument("--corpus-root", default="/corpus")
    ingest.add_argument("--reset", action="store_true")
    ingest.add_argument("--no-vision", action="store_true")
    ingest.add_argument("--parse-only", action="store_true")
    retry = sub.add_parser("retry-extraction-errors")
    retry.add_argument("path")
    retry.add_argument("--source-locator", required=True)
    retry.add_argument("--category")
    embed = sub.add_parser("embed-evidence")
    embed.add_argument("--limit", type=int, default=0)
    embed.add_argument("--concurrency", type=int, default=3)
    full = sub.add_parser("ingest-corpus")
    full.add_argument("corpus_root", nargs="?", default="/corpus")
    full.add_argument("--mode", choices=["fast-full"], default="fast-full")
    full.add_argument("--deadline-hours", type=float, default=12)
    full.add_argument("--resume", dest="run_id")
    full.add_argument("--benchmark-limit", type=int, default=0)
    status = sub.add_parser("pipeline-status")
    status.add_argument("--run-id")
    benchmark = sub.add_parser("benchmark-parser")
    benchmark.add_argument("corpus_root", nargs="?", default="/corpus")
    benchmark.add_argument("--limit", type=int, default=100)
    benchmark.add_argument("--workers", type=int, default=4)
    sub.add_parser("qa")
    sub.add_parser("initialize")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--local-only", action="store_true")
    return result


async def main_async(args: argparse.Namespace) -> dict:
    if args.command == "benchmark-parser":
        return await benchmark_fast_parser(
            args.corpus_root, limit=max(1, args.limit), workers=max(1, args.workers)
        )
    if args.command == "pipeline-status":
        store = SearchStore()
        try:
            return await asyncio.to_thread(store.status, args.run_id)
        finally:
            store.close()
    if args.command == "ingest-corpus":
        pipeline = FullCorpusPipeline()
        try:
            return await pipeline.run(
                args.corpus_root,
                deadline_hours=max(0.5, args.deadline_hours),
                run_id=args.run_id,
                benchmark_limit=max(0, args.benchmark_limit),
            )
        finally:
            pipeline.repository.close()
            pipeline.search.close()
    service = MEKGService()
    try:
        if args.command == "initialize":
            return service.initialize()
        if args.command == "preflight":
            return await service.preflight(remote=not args.local_only)
        if args.command == "qa":
            return service.qa()
        if args.command == "retry-extraction-errors":
            return await service.retry_extraction_errors(
                args.path,
                source_locator=args.source_locator,
                category=args.category,
            )
        if args.command == "embed-evidence":
            return await asyncio.to_thread(
                service.embed_evidence_chunks,
                limit=max(0, args.limit),
                concurrency=max(1, args.concurrency),
            )
        return await service.ingest_manifest(
            args.manifest,
            corpus_root=args.corpus_root,
            reset=args.reset,
            enable_vision=not args.no_vision,
            enable_extraction=not args.parse_only,
        )
    finally:
        service.repository.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args()
    result = asyncio.run(main_async(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
