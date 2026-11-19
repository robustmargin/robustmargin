"""Make the shared clustered-data builders and oracles importable.

``tests/unit/_cluster_support.py`` holds the clustered-data generators and the numerical
oracles used by the unit, property and coverage tests alike.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UNIT = str(Path(__file__).resolve().parent.parent / "unit")
if _UNIT not in sys.path:
    sys.path.insert(0, _UNIT)
