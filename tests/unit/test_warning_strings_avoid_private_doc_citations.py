"""User-facing strings must not cite documents that are not in the public repository.

The project's own working documents (decision records, the implementation plan, the statistics and
review notes) are not shipped with the package: a consumer who reads a warning or exception
message naming a decision record, a plan section or one of those notes has no way to look any of
them up. This file scans for such citations in the runtime string literals across
``src/robustmargin/*.py`` (warning messages built with ``warnings.append``/
``collected_warnings.append``, and ``raise ValueError(...)`` messages).

Two complementary checks:

1. A static scan over every non-docstring string literal in ``src/robustmargin/*.py`` (the
   authoritative, complete check: it cannot miss a reachable-but-untested code path, unlike any
   dynamic exercise). Docstrings are identified structurally (the first statement of a module,
   class or function body) and are out of scope here -- they document *why* a rule exists for a
   contributor reading the source, which is a different audience from someone reading a message a
   running program printed. ``test_tracked_text_is_self_contained.py`` scans docstrings and
   comments too.
2. A dynamic exercise of a representative sample of the real warning- and exception-producing
   paths, through the public API (plus ``robustmargin.testing``'s fakes and direct, public
   dataclass construction for paths that need a specific, otherwise hard-to-reach ``Fit``/
   ``Threshold`` shape), collecting the actual message strings a caller would see and checking
   them against the same pattern. This is not exhaustive -- the static scan is -- but it is
   direct evidence that real, user-visible text is affected, not just unreachable dead code.

The ``UPPER_CASE: `` tokens themselves (``NON_MONOTONE_DATA: ``, ``NON_MONOTONE: ``,
``OVERDISPERSED: ``, ``SPARSE_CELLS: ``) are frozen API and are pinned here too, so a rewording of
the prose cannot accidentally touch them: only the text after the token may change.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from typing import Any, cast

import pytest

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Covariance,
    Definition,
    Fit,
    Observations,
    Parameter,
    Scorecard,
    Status,
    diagnose,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.report import from_dict, to_dict
from robustmargin.testing import fake_fit, fake_threshold

_FORBIDDEN_DOC_CITATION = re.compile(
    r"decisions/\d{4}|plan section|plan §|§\d|IMPLEMENTATION_PLAN|STATISTICS\.md|REVIEWS\.md"
    r"|\bD[1-7]\b"
)

_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "robustmargin"

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_LEVEL = 0.95

# Every leading `UPPER_CASE: ` warning token found in src/robustmargin (grep -rnoE
# '[A-Z][A-Z_]{2,}: ' src/robustmargin/*.py, filtered to actual message prefixes rather than
# constant/dict-key names) -- frozen API, pinned byte-for-byte.
_KNOWN_TOKENS: tuple[str, ...] = (
    "NON_MONOTONE_DATA: ",
    "NON_MONOTONE: ",
    "OVERDISPERSED: ",
    "SPARSE_CELLS: ",
)


# ------------------------------------------------------------------------------------------
# 1. Static scan: every non-docstring string literal in src/robustmargin/*.py.
# ------------------------------------------------------------------------------------------


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """The ``id()`` of every ``ast.Constant`` that is a module/class/function docstring --
    i.e. the first statement of that scope's body, exactly the condition that makes Python
    itself treat it as ``__doc__``. Walking a function-within-a-function also finds nested
    docstrings, which this codebase uses (local helper functions inside ``_fit_generic`` and
    similar)."""
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_ids.add(id(body[0].value))
    return doc_ids


def _scan_for_doc_citations() -> list[tuple[str, int, str]]:
    """Every non-docstring string constant (including each literal fragment of an f-string,
    which ``ast`` represents as its own ``Constant`` inside a ``JoinedStr``) in
    ``src/robustmargin/*.py`` whose text matches ``_FORBIDDEN_DOC_CITATION``.

    Returns a list of ``(filename, lineno, snippet)`` so a failure is immediately actionable.
    """
    violations: list[tuple[str, int, str]] = []
    for path in sorted(_SRC_DIR.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        doc_ids = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in doc_ids:
                continue
            if _FORBIDDEN_DOC_CITATION.search(node.value):
                violations.append((path.name, node.lineno, node.value[:120]))
    return violations


def test_the_static_scan_reads_the_package_sources_it_claims_to() -> None:
    """A guard on ``_SRC_DIR``: ``Path.glob`` on a directory that does not exist yields nothing,
    so a package rename that left ``_SRC_DIR`` pointing at the old name would make the scan
    below pass vacuously. Every module that produces user-visible messages must be in it."""
    scanned = {path.name for path in _SRC_DIR.glob("*.py")}

    assert _SRC_DIR.name == "robustmargin"
    assert {"report.py", "models.py", "threshold.py", "ratio.py", "validation.py"} <= scanned


def test_no_runtime_string_literal_cites_an_unpublished_document() -> None:
    violations = _scan_for_doc_citations()
    assert not violations, (
        "runtime string literal(s) in src/robustmargin cite a document that is not in the public "
        f"repository ({_FORBIDDEN_DOC_CITATION.pattern}):\n"
        + "\n".join(f"  {name}:{lineno}: {snippet!r}" for name, lineno, snippet in violations)
    )


def test_the_scan_itself_catches_a_known_bad_literal() -> None:
    """A guard on the scanner, not on ``src/``: if ``_scan_for_doc_citations`` ever stopped
    matching anything (for example after an unrelated refactor of its own regex), the test above
    would pass vacuously. A tiny synthetic module with one offending literal and one docstring
    proves the scanner still tells them apart."""
    import tempfile

    code = (
        '"""A perfectly fine module docstring citing decisions/9999, which must be ignored."""\n'
        "\n"
        "def f():\n"
        '    """Also fine: decisions/9999 in a function docstring."""\n'
        '    return "see decisions/9999 for details"\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        module_path = Path(tmp) / "fake_module.py"
        module_path.write_text(code)
        tree = ast.parse(code, filename=str(module_path))
        doc_ids = _docstring_node_ids(tree)
        hits = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
            and _FORBIDDEN_DOC_CITATION.search(node.value)
        ]
    assert hits == ["see decisions/9999 for details"]


# ------------------------------------------------------------------------------------------
# 2. Dynamic exercise: real warning/exception paths through the public API.
# ------------------------------------------------------------------------------------------


def _wrong_sign_fit() -> Fit:
    """The wrong-sign case: success rate rises with severity, the opposite of
    ``direction="decreasing"``, with overlap at every level (no separation) -- ``NOT_CONVERGED``
    with the wrong-sign wording."""
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=[0.1, 1.0, 10.0],
        successes=[10, 50, 90],
        trials=[100, 100, 100],
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _control_incompatible_fit() -> Fit:
    """A fixed ``upper=1.0`` with a failing zero-severity control, on a log axis: the likelihood
    is zero, so ``fit_dose_response`` reports ``CONTROL_INCOMPATIBLE`` before fitting at all."""
    obs = Observations.from_counts(
        _LOG_AXIS,
        severity=[0.0, 0.01, 0.05, 0.1],
        successes=[98, 0, 0, 0],
        trials=[200, 300, 300, 300],
        outcome="success",
        direction="decreasing",
    )
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _non_monotone_converged_fit_and_threshold() -> tuple[Fit, object]:
    """The ``form_b`` `selenium` curve (README.md's own worked example, also executed by
    ``tests/unit/test_readme_examples.py``): survival rises from severity 300 to 400 before
    falling again. At ``target=0.5`` the exact one-sided bounds at severity 300 and 400
    contradict monotonicity (`tests/unit/test_threshold_status_coverage.py::
    TestNonMonotoneContradictionWarningOnAConvergedFit` pins the premise), which is what makes
    this fit produce *both* the ``NON_MONOTONE_DATA: `` (data-only, from ``Fit.monotonicity``)
    and ``NON_MONOTONE: `` (exact-bound contradiction, from ``censoring.classify``) tokens at
    once, on a converged fit."""
    conc = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    total = [141, 153, 142, 139, 154, 155]
    dead = [2, 30, 59, 82, 62, 85]
    successes = [n - d for n, d in zip(total, dead, strict=True)]
    obs = Observations.from_counts(
        Axis(name="concentration", unit="ug/L", scale="log"),
        severity=conc,
        successes=successes,
        trials=total,
        outcome="survival",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    t = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL)
    return fit, t


def _overdispersed_fit() -> Fit:
    """Three fixed-asymptote (``upper=0.75``, ``lower=0.25``) cells at exact ``z`` values
    (``-50, 0, +50``), with the middle cell's observed count chosen so the Pearson dispersion is
    strictly above the 1.5 threshold -- the same construction `tests/unit/
    test_validation.py::TestDispersionWarning` uses, independently reproduced here."""
    from scipy.special import gammaln, xlogy
    from scipy.stats import norm

    mu, s, upper, lower = 0.0, 1.0, 0.75, 0.25
    severities = [0.0, 1.0, 2.0]
    cells = (
        Cell(severity=0.0, successes=75, trials=100),
        Cell(severity=1.0, successes=16, trials=24),  # pearson term = (16-12)^2/6 = 8/3 > 1.5
        Cell(severity=2.0, successes=25, trials=100),
    )
    log_likelihood = 0.0
    for cell, severity in zip(cells, severities, strict=True):
        z = (severity - mu) / s
        p = float(upper - (upper - lower) * norm.cdf(z))
        y, n = float(cell.successes), float(cell.trials)
        lchoose = float(gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0))
        log_likelihood += lchoose + float(xlogy(y, p)) + float(xlogy(n - y, 1.0 - p))
    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=upper, fixed=True),
        "lower": Parameter(value=lower, fixed=True),
    }
    covariance = Covariance(names=("mu", "s"), matrix=((0.01, 0.0), (0.0, 0.01)))
    return Fit(
        axis=Axis(name="severity", unit="unit", scale="linear"),
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=params,
        covariance=covariance,
        log_likelihood=log_likelihood,
        cells=cells,
        status=Status.OK,
    )


def _sparse_cells_fit() -> Fit:
    """A fixed ``upper=1, lower=0`` fit with one cell whose expected-failure count is below the
    0.1 cutoff with exactly one observed failure -- the same construction `tests/unit/
    test_validation.py::TestSparseCellsWarning` uses."""
    from scipy.special import gammaln, xlogy
    from scipy.stats import norm

    mu, s, upper, lower = 0.6931471805599453, 1.0, 1.0, 0.0  # mu = log(2)
    # Ascending in severity, as ``Fit.cells`` must be: 0.05 has no observed failure, 0.09 is the
    # flagged cell, 0.1522 is above the cutoff.
    severities = [0.05, 0.09, 0.1522]
    successes_ = [100, 99, 99]
    trials_ = [100, 100, 100]
    cells = tuple(
        Cell(severity=sv, successes=y, trials=n)
        for sv, y, n in zip(severities, successes_, trials_, strict=True)
    )
    log_likelihood = 0.0
    for cell, severity in zip(cells, severities, strict=True):
        z = (__import__("math").log(severity) - mu) / s
        p = float(upper - (upper - lower) * norm.cdf(z))
        y, n = float(cell.successes), float(cell.trials)
        lchoose = float(gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0))
        log_likelihood += lchoose + float(xlogy(y, p)) + float(xlogy(n - y, 1.0 - p))
    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=upper, fixed=True),
        "lower": Parameter(value=lower, fixed=True),
    }
    covariance = Covariance(names=("mu", "s"), matrix=((0.01, 0.0), (0.0, 0.01)))
    return Fit(
        axis=_LOG_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=params,
        covariance=covariance,
        log_likelihood=log_likelihood,
        cells=cells,
        status=Status.OK,
    )


def _collect_dynamic_messages() -> list[str]:
    """Every message string from a representative sample of real warning- and
    exception-producing paths, reached through the public API (plus ``robustmargin.testing``'s
    fakes, and direct public-dataclass construction for shapes a real fit cannot easily be
    coaxed into). Not exhaustive -- the static scan above is."""
    messages: list[str] = []

    messages.extend(_wrong_sign_fit().warnings)
    messages.extend(_control_incompatible_fit().warnings)

    non_monotone_fit, non_monotone_threshold = _non_monotone_converged_fit_and_threshold()
    messages.extend(non_monotone_fit.monotonicity.warnings if non_monotone_fit.monotonicity else ())
    messages.extend(non_monotone_threshold.warnings)

    messages.extend(diagnose(_overdispersed_fit()).warnings)
    messages.extend(diagnose(_sparse_cells_fit()).warnings)

    # ratio_interval: dependence='paired' is not implemented in this release.
    t_a = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
    t_b = fake_threshold(status=Status.OK, censoring=Censoring.NONE)
    with pytest.raises(ValueError) as excinfo_paired:
        ratio_interval(t_a, t_b, method="fieller", dependence="paired")
    messages.append(str(excinfo_paired.value))

    # ratio_interval: an explicit level that disagrees with both thresholds' own.
    with pytest.raises(ValueError) as excinfo_level:
        ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=0.80)
    messages.append(str(excinfo_level.value))

    # threshold(): definition.kind='relative' on a failed fit with no zero-severity control cell
    # to fall back to.
    no_control_fit = Fit(
        axis=_LOG_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=(Cell(severity=1.0, successes=50, trials=100),),
        status=Status.NOT_CONVERGED,
    )
    with pytest.raises(ValueError) as excinfo_relative:
        threshold(
            no_control_fit,
            definition=Definition.relative(0.5),
            interval_method="profile",
            level=_LEVEL,
        )
    messages.append(str(excinfo_relative.value))

    # from_dict: a card tagged with a schema_version that is not recognised -- once on a whole
    # card, once nested inside a current one. The message is read by a person holding a card from
    # elsewhere, which is exactly the reader who cannot look up a document, so it is held to the
    # same rule as every other message here.
    for unrecognised_version in ("0", "3", "999"):
        unrecognised = to_dict(fake_threshold())
        unrecognised["schema_version"] = unrecognised_version
        with pytest.raises(ValueError) as excinfo_unrecognised:
            from_dict(unrecognised)
        messages.append(str(excinfo_unrecognised.value))

    nested_card: dict[str, Any] = to_dict(Scorecard(results=(fake_threshold(),), provenance={}))
    nested_card["results"][0]["fit"]["schema_version"] = "3"
    with pytest.raises(ValueError) as excinfo_nested:
        from_dict(nested_card)
    messages.append(str(excinfo_nested.value))

    return messages


def _collect_cell_ordering_messages() -> list[str]:
    """The ``Fit.cells`` ordering errors: the construction error for a
    descending pair and for a duplicated severity, and the same error reached through
    ``from_dict`` on a card whose cells were reversed by hand. Each is read by a person holding a
    hand-built fit or a hand-edited card, who cannot look up a document either. Kept apart from
    ``_collect_dynamic_messages`` so the token-prefix tests below do not depend on it."""
    messages: list[str] = []
    fit = fake_fit()
    # Never touch the zero-severity cell's counts: a failing control under the fake's fixed
    # upper=1.0 is rejected by a different rule, which would put another message here.
    control, level_one = fit.cells[0], fit.cells[1]

    with pytest.raises(ValueError) as excinfo_descending:
        dataclasses.replace(fit, cells=tuple(reversed(fit.cells)))
    messages.append(str(excinfo_descending.value))

    repeated = dataclasses.replace(level_one, successes=level_one.successes - 1)
    with pytest.raises(ValueError) as excinfo_duplicate:
        dataclasses.replace(fit, cells=(control, level_one, repeated))
    messages.append(str(excinfo_duplicate.value))

    reversed_card = cast("dict[str, Any]", to_dict(fit))
    reversed_card["cells"] = list(reversed(reversed_card["cells"]))
    with pytest.raises(ValueError) as excinfo_reversed_card:
        from_dict(reversed_card)
    messages.append(str(excinfo_reversed_card.value))

    return messages


def test_cell_ordering_errors_do_not_cite_an_unpublished_document() -> None:
    messages = _collect_cell_ordering_messages()
    assert len(messages) == 3, "the three ordering errors were not all produced -- a fixture broke"

    offending = [m for m in messages if _FORBIDDEN_DOC_CITATION.search(m)]
    assert not offending, (
        "Fit.cells ordering error message(s) cite a document that is not in the public "
        "repository:\n" + "\n".join(f"  {m!r}" for m in offending)
    )


def test_dynamic_warning_and_exception_paths_do_not_cite_an_unpublished_document() -> None:
    messages = _collect_dynamic_messages()
    assert messages, "the dynamic exercise above produced no messages at all -- a fixture broke"

    offending = [m for m in messages if _FORBIDDEN_DOC_CITATION.search(m)]
    assert not offending, (
        "real, user-visible warning/exception message(s) cite a document that is not in the "
        "public repository:\n" + "\n".join(f"  {m!r}" for m in offending)
    )


# ------------------------------------------------------------------------------------------
# 3. The UPPER_CASE: tokens themselves are frozen API -- pinned so a prose fix cannot drift them.
# ------------------------------------------------------------------------------------------


def test_every_known_token_is_produced_by_some_real_path() -> None:
    """A guard on this file's own fixtures: if a future refactor silently stopped a fixture from
    producing its token, the token-prefix tests below would never run against real text."""
    messages = _collect_dynamic_messages()
    for token in _KNOWN_TOKENS:
        assert any(m.startswith(token) for m in messages), (
            f"no message among the dynamic exercise's own fixtures starts with {token!r}; "
            "this file's own fixtures need revisiting"
        )


@pytest.mark.parametrize("token", _KNOWN_TOKENS)
def test_known_token_is_a_byte_identical_prefix(token: str) -> None:
    messages = _collect_dynamic_messages()
    matching = [m for m in messages if m.startswith(token)]
    assert matching, f"no dynamically-produced message starts with {token!r}"
    for message in matching:
        assert message[: len(token)] == token
