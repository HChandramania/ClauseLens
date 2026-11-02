#!/usr/bin/env python3
"""Build or reuse the dense passage cache for a pipeline configuration.

    python scripts/build_dense_index.py --config configs/serve.yaml

The command indexes the same CUAD-mini documents used by the CLI and app. Dense
retrieval caches are reused automatically when the model and chunk IDs match.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import load_config
from src.data import load_mini_cuad
from src.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.retrieval.name != "dense":
        raise SystemExit("dense index building requires a config with retrieval.name: dense")

    _, documents = load_mini_cuad(mode=config.data.read_mode, limit=config.data.mini_limit)
    pipeline = Pipeline.build(config)
    retriever = pipeline.retriever
    cache_path = getattr(retriever, "cache_path", None)
    print(f"documents={len(documents)}", flush=True)
    print(f"cache_path={cache_path}", flush=True)
    started = time.perf_counter()
    stats = pipeline.index(documents.values())
    elapsed = time.perf_counter() - started
    print(f"chunks={stats.chunks} index_seconds={elapsed:.2f}", flush=True)
    print(f"cache_hit={getattr(retriever, 'cache_hit', False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
