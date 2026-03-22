"""conftest.py — Add repo root to sys.path for pytest.

This ensures `import app.jobs`, `import app.app` etc. resolve correctly
when running pytest from the repo root.
"""
import sys
from pathlib import Path

# Add repo root so `import app.jobs`, `import app.app` etc. resolve correctly.
repo_root = str(Path(__file__).parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
