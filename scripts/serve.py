#!/usr/bin/env python3
"""Run the Kenshō API server.

    python scripts/serve.py
    KENSHO_ALLOW_FALLBACK=true python scripts/serve.py   # demo mode, no models needed

Configuration is entirely environment-driven (see ``serve/config.py``) so the
same command works locally and in a container — set ``GROQ_API_KEY`` and
point ``KENSHO_INDEX_DIR`` at a built index for the real thing, or set
``KENSHO_ALLOW_FALLBACK=true`` with no other configuration to smoke-test the
service with fake/echo components and no external dependencies at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from kensho.serve.app import create_app  # noqa: E402
from kensho.serve.config import ServerConfig  # noqa: E402


def main() -> int:
    config = ServerConfig.from_env()
    print(f"embedder={config.embedder_model or 'fake'}  llm={config.llm_provider or 'echo'}  "
         f"verifier={config.verifier_method}  index_dir={config.index_dir}  "
         f"allow_fallback={config.allow_fallback}")
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec - intended for container use
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
