"""Import hygiene: importing ``robustmargin`` must not drag in banned dependencies.

robustmargin forbids any GPU, ML-framework, or plotting dependency, at any time, for any
reason. The only way to check that honestly is to import the real package in a
fresh interpreter and inspect ``sys.modules`` there — asserting against ``sys.modules`` inside
the running pytest process is worthless, since pytest and unrelated test modules have already
imported arbitrary things by the time any test body runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Visible in one place so it can be extended as new forbidden dependencies are named.
# `pandas` is deliberately absent: `statsmodels` requires pandas, so its presence in
# `sys.modules` after importing `robustmargin` is unavoidable and is not a constraint violation.
# Do not "tighten" this test by adding it.
FORBIDDEN_MODULES: tuple[str, ...] = ("torch", "tensorflow", "jax", "matplotlib")

_SUBPROCESS_SCRIPT = """
import sys

import robustmargin  # noqa: F401

forbidden = {forbidden!r}
leaked = sorted(name for name in forbidden if name in sys.modules)
print(",".join(leaked))
"""


def _run_import_in_subprocess() -> tuple[int, str, str]:
    """Import ``robustmargin`` in a fresh interpreter and report which forbidden modules leaked.

    Returns
    -------
    tuple[int, str, str]
        The subprocess return code, its stdout, and its stderr.
    """
    script = _SUBPROCESS_SCRIPT.format(forbidden=FORBIDDEN_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout, result.stderr


def test_importing_robustmargin_does_not_load_forbidden_modules() -> None:
    """A plain ``import robustmargin`` must never pull torch, tensorflow, jax, or matplotlib."""
    returncode, stdout, stderr = _run_import_in_subprocess()
    assert returncode == 0, f"subprocess failed to import robustmargin: {stderr}"

    leaked = [name for name in stdout.strip().split(",") if name]
    assert not leaked, (
        f"forbidden module(s) leaked into sys.modules on import of robustmargin: {leaked}"
    )


def test_robustmargin_version_is_a_nonempty_string() -> None:
    """``robustmargin.__version__`` must exist and be a non-empty string.

    The exact value is not asserted: it changes every release, and pinning it here would make
    this test a maintenance tax rather than a hygiene check.
    """
    import robustmargin

    assert isinstance(robustmargin.__version__, str)
    assert robustmargin.__version__ != ""


# robustmargin.report and robustmargin.testing are importable submodules, not re-exported from
# robustmargin/__init__.py. Neither may pull in a dev-only dependency merely by being imported --
# jsonschema is a dev extra used by robustmargin.testing's *own* test suite and by consumers'
# validators, not by loading the module itself; hypothesis and pytest belong to robustmargin's own
# test suite. A production install of a consumer that only imports these modules must not need
# any of the three.
_DEV_ONLY_MODULES: tuple[str, ...] = ("jsonschema", "hypothesis", "pytest")

_REPORT_AND_TESTING_SUBPROCESS_SCRIPT = """
import sys

import robustmargin  # noqa: F401
import robustmargin.report  # noqa: F401
import robustmargin.testing  # noqa: F401

forbidden = {forbidden!r}
leaked = sorted(name for name in forbidden if name in sys.modules)
print(",".join(leaked))
"""


def test_importing_report_and_testing_does_not_load_dev_only_modules() -> None:
    """A fresh-interpreter import of ``robustmargin``, ``robustmargin.report`` and
    ``robustmargin.testing`` together must never pull in ``jsonschema``, ``hypothesis`` or
    ``pytest``."""
    script = _REPORT_AND_TESTING_SUBPROCESS_SCRIPT.format(forbidden=_DEV_ONLY_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed to import robustmargin: {result.stderr}"

    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        f"dev-only module(s) leaked into sys.modules on import of robustmargin/report/testing: "
        f"{leaked}"
    )


# The source tree holds one importable package. Nothing else ships -- no second import package, no
# alias module, no re-export -- because a top-level package inside this wheel would overwrite, or
# be overwritten by, whatever another project publishes under that name.
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def test_src_holds_exactly_the_robustmargin_package() -> None:
    """The source tree has one importable package, ``robustmargin``, and no top-level module.
    Judged from the files on disk rather than from an import failing, because a development
    environment may carry a stale editable install of another name."""
    packages = sorted(p.name for p in _SRC_DIR.iterdir() if (p / "__init__.py").is_file())
    top_level_modules = sorted(p.name for p in _SRC_DIR.glob("*.py"))

    assert packages == ["robustmargin"]
    assert top_level_modules == []
