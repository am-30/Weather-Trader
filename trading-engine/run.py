"""
Workflow entry point.

This file is the shell target for the Replit workflow.
It adds the ``trading-engine`` directory to ``sys.path`` so that
``import src.*`` works correctly, then delegates to ``src.main.main()``.
"""

from __future__ import annotations

import os
import sys

# Ensure imports resolve relative to this file's directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.main import main  # noqa: E402

if __name__ == "__main__":
    main()
