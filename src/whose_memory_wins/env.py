"""Process-wide environment setup. Imported before mem0/fastembed are loaded."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Never send Mem0 usage telemetry from research runs.
os.environ["MEM0_TELEMETRY"] = "False"
# Keep the local embedding model inside the project (ignored by git) instead of %TEMP%.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(PROJECT_ROOT / ".cache" / "fastembed"))
# Mem0 keeps a small config dir; keep it inside the project too.
os.environ.setdefault("MEM0_DIR", str(PROJECT_ROOT / ".cache" / "mem0"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass
