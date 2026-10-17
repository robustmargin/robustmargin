"""Executes every fenced ```python block in ``README.md``, so its examples cannot silently rot
as the public API changes underneath them (any ```python block added to the README later is picked
up automatically, with no test change needed).

Each block is executed in its own fresh namespace (not accumulated top-to-bottom): every block
in the README already re-imports what it needs, so this does not assume any cross-block reading
order and catches a block that silently depended on a name defined earlier by accident.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_README_PATH = Path(__file__).resolve().parents[2] / "README.md"
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks() -> list[str]:
    text = _README_PATH.read_text(encoding="utf-8")
    return _PYTHON_BLOCK_RE.findall(text)


_BLOCKS = _extract_python_blocks()


def test_readme_has_at_least_one_python_block() -> None:
    """A guard on the extraction itself: a regex that silently stopped matching (for example
    after a README reformat) would otherwise make every block "test" below vacuously disappear
    rather than fail."""
    assert len(_BLOCKS) >= 1


@pytest.mark.parametrize("index", range(len(_BLOCKS)))
def test_readme_python_block_executes_without_error(index: int) -> None:
    code = _BLOCKS[index]
    exec(compile(code, f"README.md:python-block-{index}", "exec"), {"__name__": "__main__"})
