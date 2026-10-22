"""Every tracked text file must make sense to a reader who has only the repository.

``robustmargin`` is a public repository and its releases are permanent. The project's own working
documents (decision records, the implementation plan, the roadmap, the review log, the consumer
contract, the architecture and statistics notes, the validation notes, the agent-instruction file)
are gitignored and never published. A docstring, comment, schema description, workflow note or
README sentence that cites one of them -- a record number, "plan section 5.4", a section sign,
"hard constraint 3", an owner decision, a decision label such as ``D5``, "Phase 7", a requirement
label such as "(R2)", a review-finding label, an agent's name -- sends the reader to a document
they cannot open. The repository also does not narrate the project's history: no former package
name, no name of the repository a function came from, no earlier release or schema number.

This file scans the **raw text of every git-tracked file** (``git ls-files``), so docstrings,
comments, workflow files, test data and reference-fixture scripts are all covered. Files that do
not decode as UTF-8 are skipped, as is ``LICENSE``, whose Apache text legitimately says
"copyright owner". The two guard test files are the other exclusion: they must hold the patterns
themselves, so they are scanned for the retired names only (and those are assembled from fragments
below, so the literal never appears in the tree).

Pattern groups, reported separately so a failure says which kind of leak it is:

``_CITATION_PATTERNS``
    The agreed list of private-document references.
``_DANGLING_PATTERNS``
    References that only mean something *inside* a private document -- "Amendment 1", "item 6",
    a bare "section 5.6", "stats-review fix", "design choice 2", a review "round 3", a numbered
    "rule 12". They are what is left behind when the document's name is removed but its internal
    numbering is not.
``_HISTORY_PATTERNS``
    The retired names (the former package name, and the source repository's name, bare or in
    full), the removed attribution file's name, the pre-release version strings, and the
    retired series labels of the source domain. A future schema file such as
    ``scorecard-v2.json`` is legitimate, so schema numbers are not patterns here.
``_SHIPPED_ONLY_PATTERNS``
    The strict ``owner`` rule. It applies to the files that ship (the package sources and schema,
    the PyPI page, the sdist metadata). Elsewhere "owner" is ordinary vocabulary -- the code owner
    in ``CODEOWNERS`` and in the pull-request template's closing line, PyPI's own term for a
    project's GitHub organization -- and only decision-style uses are forbidden
    (``_OWNER_PATTERNS``).

One path exemption: ``.gitignore`` has to name the private paths in order to ignore them, so it is
exempt from the private-document-name and ``docs/`` patterns, and from nothing else.

The scan is per file and each failure lists *every* offending ``file:line`` with the matched text,
so one run is a complete worklist.

A scanner that matches nothing would pass vacuously, so the matcher itself is pinned: known-bad
strings are caught, known-good strings (R-language references to ``drc`` and ``glm``, "planned",
"planning", numeric literals such as ``0.0025``, literature citations, the lowercase ``provenance``
result field) are not, and the file list is checked to be non-trivial.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# (label, pattern). The label is what a failure prints next to the matched text.
_Pattern = tuple[str, re.Pattern[str]]

# The retired names are never written literally, here or anywhere else in the tree: they are
# assembled from fragments, and this file is itself scanned for them.
_FORMER_PACKAGE = "margin" + "kit"
_SOURCE_REPO = "ze" + "ta" + "-bench"

_NAME_PATTERNS: tuple[_Pattern, ...] = (
    # The former package name, with or without a separator, in any case.
    ("retired package name", re.compile("margin" + r"[\W_]?" + "kit", re.IGNORECASE)),
    # The source repository, and its short form on its own ("the <short form> matrix", a data file
    # or test module named after it). No other use of this word belongs in this package.
    ("retired source-repository name", re.compile("ze" + "ta", re.IGNORECASE)),
)

_CITATION_PATTERNS: tuple[_Pattern, ...] = (
    ("decision-record path", re.compile(r"decisions/")),
    # A bare decision-record number: four digits with a leading zero, used as a citation
    # (`0023`, `0023 Amendment 2`, "0008, 0009"). It must not match a digit run that is part of a
    # number: `0.0025` (preceded by "."), `1e-0025` (preceded by an exponent sign), `10025` or
    # `1_0025` (preceded by a word character), `0.00250` or `0023.5` (followed by a word
    # character or by "." and a digit). The four digit groups of an ORCID iD URL
    # (`https://orcid.org/0009-0005-3586-7901`) are exempt: that is a public researcher
    # identifier, not a citation. The exemption is anchored on `orcid.org/` (fixed-width
    # look-behinds, one per group), so a bare `0009-0005` or `0031-0034` is still flagged.
    (
        "decision-record number",
        re.compile(
            r"(?<![\w.])(?<![eE][-+])"
            r"(?<!orcid\.org/)(?<!orcid\.org/\d{4}-)"
            r"(?<!orcid\.org/\d{4}-\d{4}-)(?<!orcid\.org/\d{4}-\d{4}-\d{4}-)"
            r"0\d{3}(?!\w|\.\d)"
        ),
    ),
    ("section sign", re.compile("§")),
    # The word "plan" and its plural; `\b` keeps "planned", "planning", "planar", "explanation".
    ("plan", re.compile(r"\bplans?\b", re.IGNORECASE)),
    (
        "private document name",
        re.compile(
            r"IMPLEMENTATION_PLAN|\bROADMAP\b|\bPITCH\b"
            r"|\b(?:REVIEWS|CONSUMERS|ARCHITECTURE|STATISTICS|CLAUDE)\b(?:\.md)?"
            r"|validation\.md"
        ),
    ),
    ("hard constraint", re.compile(r"\bhard[- ]constraints?\b", re.IGNORECASE)),
    ("decision label D1-D7", re.compile(r"\bD[1-7]\b")),
    ("phase number", re.compile(r"\bphases? \d", re.IGNORECASE)),
    ("Appendix B", re.compile(r"\bAppendix B\b")),
    # R1-R10 requirement labels and "R6 #14" review references. "R `drc`", "R's glm" and
    # "R-squared" carry no digit directly after the R, so they do not match.
    ("requirement label R1-R10", re.compile(r"\bR(?:10|[1-9])\b")),
    # Review-finding labels ("finding C2", "W1", "(S3)"). Restricted to the letters actually used
    # for findings so that a future "H0" or "L2" in a statistical statement does not trip it. The
    # trailing look-ahead keeps chemical-style tokens ("C2H5") and decimals ("S3.5") out.
    ("review-finding label", re.compile(r"(?<![\w.])[ACSW]\d{1,2}(?![\w.])")),
    ("agent name", re.compile(r"\b(?:test-author|api-compat|stats-reviewer|implementer)\b")),
    # Any `docs/` path. `readthedocs/`-style host or path segments ending in "docs" are not
    # `docs/` directories.
    ("docs/ path", re.compile(r"(?<![\w-])docs/")),
)

# "owner" outside the shipped files: only the decision-style uses, which cite an owner's ruling
# that the reader cannot see. "the code owner", "the owner's review" and "requires the owner's
# approval" are ordinary descriptions of a person or role and stay legal.
_OWNER_PATTERNS: tuple[_Pattern, ...] = (
    (
        "owner decision (decision-style)",
        re.compile(r"\bowner(?:'s)? decisions?\b|\bowner-approved\b|\(\s*owner\b", re.IGNORECASE),
    ),
)

_DANGLING_PATTERNS: tuple[_Pattern, ...] = (
    ("Amendment", re.compile(r"\bamendments?\b", re.IGNORECASE)),
    ("bare section number", re.compile(r"\bsections? \d", re.IGNORECASE)),
    ("bare item number", re.compile(r"\bitems? \d", re.IGNORECASE)),
    ("stats review", re.compile(r"\bstats[- ]review", re.IGNORECASE)),
    ("design choice number", re.compile(r"\bdesign choices? \d", re.IGNORECASE)),
    # A review round ("Item 3, round 3", "Round 4, section 1"): the numbering belongs to a review
    # log the reader cannot open.
    ("review round", re.compile(r"\brounds? \d", re.IGNORECASE)),
    # A numbered rule ("Rule 12", "rules 1-3"): the numbering is that of a private list of
    # invariants. The invariant itself is stated in place instead.
    ("numbered rule", re.compile(r"\brules? \d", re.IGNORECASE)),
)

_HISTORY_PATTERNS: tuple[_Pattern, ...] = (
    *_NAME_PATTERNS,
    # The removed attribution file. Uppercase only: lowercase `provenance` is the public result
    # field on every type.
    ("removed attribution file", re.compile(r"PROVENANCE")),
    # Version strings of the pre-release line. `(?<![\d.])` rather than `\b` so a tag such as
    # `v0.1.0a1` still matches, while `10.1.0` and `1.0.1.0` do not.
    ("pre-release 0.1.0aN", re.compile(r"(?<![\d.])0\.1\.0a\d+")),
    ("release 0.1.0", re.compile(r"(?<![\d.])0\.1\.0(?!\d|\.\d)")),
    ("pre-release 0.1.1rc1", re.compile(r"(?<![\d.])0\.1\.1rc\d+")),
    # The series labels of the source domain, which the neutral fixtures replaced.
    (
        "retired series label",
        re.compile(r"(?<![A-Za-z0-9])(?:pid|ppo|sac)_(?:wind|mass)(?![A-Za-z0-9])"),
    ),
)

# Scanned in every tracked file other than the exclusions below.
_ALL_FILE_PATTERNS: tuple[_Pattern, ...] = (
    *_CITATION_PATTERNS,
    *_OWNER_PATTERNS,
    *_DANGLING_PATTERNS,
    *_HISTORY_PATTERNS,
)

# Scanned in the shipped files only, in addition.
_SHIPPED_ONLY_PATTERNS: tuple[_Pattern, ...] = (
    ("owner", re.compile(r"\bowners?\b", re.IGNORECASE)),
)

# Exempt labels per path. `.gitignore` names the private paths in order to ignore them.
_EXEMPT_LABELS: dict[str, frozenset[str]] = {
    ".gitignore": frozenset({"private document name", "docs/ path"}),
}

# `LICENSE` is Apache text, which says "copyright owner" and is not ours to edit.
_EXCLUDED_FILES: frozenset[str] = frozenset({"LICENSE"})

# The guard files hold the patterns themselves, so they cannot be scanned for them; they are
# scanned for the retired names only.
_GUARD_FILES: tuple[str, ...] = (
    "tests/unit/test_tracked_text_is_self_contained.py",
    "tests/unit/test_warning_strings_avoid_private_doc_citations.py",
)


class _Hit(NamedTuple):
    lineno: int
    labels: tuple[str, ...]
    matched: tuple[str, ...]
    line: str


def _scan_text(
    text: str,
    patterns: tuple[_Pattern, ...],
    *,
    exempt_labels: frozenset[str] = frozenset(),
) -> list[_Hit]:
    """Every line of ``text`` matching any of ``patterns``, with 1-based line numbers.

    Patterns whose label is in ``exempt_labels`` are skipped.
    """
    active = tuple((label, pattern) for label, pattern in patterns if label not in exempt_labels)
    hits: list[_Hit] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        labels: list[str] = []
        matched: list[str] = []
        for label, pattern in active:
            for match in pattern.finditer(raw_line):
                labels.append(label)
                matched.append(match.group(0))
        if labels:
            hits.append(_Hit(lineno, tuple(labels), tuple(matched), raw_line.strip()))
    return hits


def _format_hits(relpath: str, hits: list[_Hit]) -> str:
    rows = []
    for hit in hits:
        found = ", ".join(f"{label}={text!r}" for label, text in zip(hit.labels, hit.matched))
        rows.append(f"  {relpath}:{hit.lineno}: {found}\n      {hit.line[:150]}")
    return "\n".join(rows)


def _tracked_paths() -> list[str]:
    """Every path ``git ls-files`` lists, relative to the repository root, sorted.

    A failure to run git raises: a guard that cannot list the tree must not pass by scanning
    nothing. A path that is listed but missing on disk (a deletion not yet staged) is dropped.
    """
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    names = completed.stdout.decode("utf-8").split("\0")
    return sorted(name for name in names if name and (_ROOT / name).is_file())


def _decode(path: Path) -> str | None:
    """The file's text, or ``None`` if it does not decode as UTF-8."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_text(relpath: str) -> str | None:
    return _decode(_ROOT / relpath)


def _is_shipped(relpath: str) -> bool:
    """The files that ship: package sources and schema, the PyPI page, the sdist metadata."""
    if relpath in ("README.md", "pyproject.toml"):
        return True
    return relpath.startswith("src/robustmargin/") and relpath.endswith((".py", ".json"))


def _patterns_for(relpath: str) -> tuple[_Pattern, ...]:
    if _is_shipped(relpath):
        return (*_ALL_FILE_PATTERNS, *_SHIPPED_ONLY_PATTERNS)
    return _ALL_FILE_PATTERNS


_TRACKED = _tracked_paths()
_GUARDS = [name for name in _TRACKED if name in _GUARD_FILES]
_SCANNED = [
    name
    for name in _TRACKED
    if name not in _EXCLUDED_FILES and name not in _GUARD_FILES and _read_text(name) is not None
]


def _scan_file(relpath: str) -> list[_Hit]:
    text = _read_text(relpath)
    assert text is not None
    return _scan_text(
        text, _patterns_for(relpath), exempt_labels=_EXEMPT_LABELS.get(relpath, frozenset())
    )


# ------------------------------------------------------------------------------------------
# 1. The scan of every tracked text file.
# ------------------------------------------------------------------------------------------

_PRIVATE_GROUPS = {label for label, _ in (*_CITATION_PATTERNS, *_OWNER_PATTERNS)} | {"owner"}
_DANGLING_LABELS = {label for label, _ in _DANGLING_PATTERNS}
_HISTORY_LABELS = {label for label, _ in _HISTORY_PATTERNS}


def _fail_if_hits(relpath: str, labels: set[str], reason: str) -> None:
    hits = [hit for hit in _scan_file(relpath) if labels & set(hit.labels)]
    # A line can match several groups; report only the matches of the group under test.
    rows = [
        hit._replace(
            labels=tuple(label for label in hit.labels if label in labels),
            matched=tuple(text for label, text in zip(hit.labels, hit.matched) if label in labels),
        )
        for hit in hits
    ]
    if rows:
        pytest.fail(
            f"{relpath} is in the public repository but {reason} ({len(rows)} line(s)):\n"
            + _format_hits(relpath, rows),
            pytrace=False,
        )


@pytest.mark.parametrize("relpath", _SCANNED)
def test_tracked_file_cites_no_unpublished_document(relpath: str) -> None:
    _fail_if_hits(
        relpath,
        _PRIVATE_GROUPS,
        "cites material that is not published; state the reason in place instead",
    )


@pytest.mark.parametrize("relpath", _SCANNED)
def test_tracked_file_has_no_dangling_reference_into_an_unpublished_document(
    relpath: str,
) -> None:
    _fail_if_hits(
        relpath,
        _DANGLING_LABELS,
        "numbers things that only exist in an unpublished document; state the rule itself "
        "instead of its number",
    )


@pytest.mark.parametrize("relpath", _SCANNED)
def test_tracked_file_states_no_project_history(relpath: str) -> None:
    _fail_if_hits(
        relpath,
        _HISTORY_LABELS,
        "names a retired name, a removed file, an earlier release, or a retired series label",
    )


@pytest.mark.parametrize("relpath", _GUARDS)
def test_guard_file_holds_no_retired_name_literally(relpath: str) -> None:
    text = _read_text(relpath)
    assert text is not None
    hits = _scan_text(text, _NAME_PATTERNS)
    if hits:
        pytest.fail(
            f"{relpath} must assemble the retired names from fragments, but spells one out "
            f"({len(hits)} line(s)):\n" + _format_hits(relpath, hits),
            pytrace=False,
        )


# ------------------------------------------------------------------------------------------
# 2. Guards on the scan itself: it must read the files it claims to, and it must be able to fail.
# ------------------------------------------------------------------------------------------


def test_tracked_file_list_comes_from_git_and_is_not_trivially_small() -> None:
    # Not vacuous: an empty or mislocated listing would make every parametrised test above
    # silently disappear.
    assert len(_TRACKED) >= 40
    for expected in (
        "README.md",
        "pyproject.toml",
        "LICENSE",
        "src/robustmargin/__init__.py",
        "src/robustmargin/report.py",
        "tools/drc_reference/generate.R",
        "tests/unit/test_tracked_text_is_self_contained.py",
        ".github/workflows/ci.yml",
    ):
        assert expected in _TRACKED, f"{expected} is not in `git ls-files`"
    assert any(
        name.startswith("src/robustmargin/schema/scorecard-v") and name.endswith(".json")
        for name in _TRACKED
    )
    assert any(name.startswith("tests/data/") for name in _TRACKED)
    assert any(
        name.startswith("tools/drc_reference/") and name.endswith(".json") for name in _TRACKED
    )


def test_scan_covers_the_package_the_workflows_the_tools_and_the_tests() -> None:
    scanned = set(_SCANNED)

    for module in (
        "__init__",
        "censoring",
        "empirical",
        "intervals",
        "models",
        "ratio",
        "report",
        "testing",
        "threshold",
        "types",
        "validation",
    ):
        assert f"src/robustmargin/{module}.py" in scanned
    assert "README.md" in scanned
    assert "pyproject.toml" in scanned
    assert ".github/workflows/ci.yml" in scanned
    assert "tools/drc_reference/generate.R" in scanned
    assert "tests/unit/test_report.py" in scanned
    assert len(_SCANNED) >= 40


def test_exclusions_are_exactly_the_license_and_the_guard_files() -> None:
    scanned = set(_SCANNED)

    assert "LICENSE" in _TRACKED
    assert "LICENSE" not in scanned
    assert _GUARDS, "the guard files are not tracked, so nothing checks them for retired names"
    assert set(_GUARDS) == set(_GUARD_FILES)
    assert not scanned & set(_GUARD_FILES)


def test_no_tracked_file_of_a_text_type_is_skipped_as_undecodable() -> None:
    """Skipping what does not decode is for binary blobs. A source, document, configuration or
    data file that stops decoding would otherwise drop out of the scan without a trace."""
    text_suffixes = (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".csv", ".R", ".txt", ".cfg")
    skipped = [
        name
        for name in _TRACKED
        if name.endswith(text_suffixes) and name not in _EXCLUDED_FILES and _read_text(name) is None
    ]

    assert skipped == []


def test_every_scanned_text_file_is_nonempty_except_the_py_typed_marker() -> None:
    for relpath in _SCANNED:
        text = _read_text(relpath)
        assert text is not None
        if relpath.endswith("py.typed"):
            continue
        assert text.strip(), f"{relpath} is tracked but empty"


def test_a_file_that_does_not_decode_as_utf8_reads_as_none_and_text_reads_as_text(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00decisions/0023")
    text = tmp_path / "note.txt"
    text.write_text("caf\u00e9 \u00a75", encoding="utf-8")

    assert _decode(binary) is None
    assert _decode(text) == "caf\u00e9 \u00a75"


def test_scan_reports_the_line_number_and_matched_text_of_every_offence() -> None:
    text = (
        '"""A fine docstring."""\n'
        "x = 1  # see decisions/0023 and plan section 5.4\n"
        "\n"
        "y = 2  # nothing wrong here\n"
        "z = 3  # hard constraint 3\n"
    )

    hits = _scan_text(text, _CITATION_PATTERNS)

    assert [hit.lineno for hit in hits] == [2, 5]
    assert "decisions/" in hits[0].matched
    assert "0023" in hits[0].matched
    assert "plan" in hits[0].matched
    assert hits[1].matched == ("hard constraint",)
    report = _format_hits("pkg/mod.py", hits)
    assert "pkg/mod.py:2:" in report
    assert "pkg/mod.py:5:" in report
    assert "'decisions/'" in report


def test_a_comment_or_docstring_is_scanned_not_only_string_literals() -> None:
    text = '"""Module docstring citing decisions/0023."""\n# a comment citing the plan\n'

    assert [hit.lineno for hit in _scan_text(text, _CITATION_PATTERNS)] == [1, 2]


def test_the_gitignore_exemption_covers_private_names_and_docs_paths_and_nothing_else() -> None:
    exempt = _EXEMPT_LABELS[".gitignore"]
    ignore_file = (
        "# Private working material.\n"
        "CLAUDE.md\n"
        ".claude/\n"
        "docs/*\n"
        "# the implementation plan, see phase 3 and decisions/0001, ROADMAP.md\n"
        f"# kept from {_FORMER_PACKAGE}\n"
    )

    hits = _scan_text(ignore_file, _ALL_FILE_PATTERNS, exempt_labels=exempt)

    # Lines 2-4 are the entries themselves; line 1 is plain prose.
    assert [hit.lineno for hit in hits] == [5, 6]
    flagged = {label for hit in hits for label in hit.labels}
    assert {"plan", "phase number", "decision-record path", "retired package name"} <= flagged
    assert "private document name" not in flagged
    assert "docs/ path" not in flagged
    # The same lines are all caught in any other file.
    assert [hit.lineno for hit in _scan_text(ignore_file, _ALL_FILE_PATTERNS)] == [2, 4, 5, 6]


def test_the_only_path_exemption_is_gitignore() -> None:
    assert set(_EXEMPT_LABELS) == {".gitignore"}


# ------------------------------------------------------------------------------------------
# 3. Pinning the matcher: known-bad strings are caught.
# ------------------------------------------------------------------------------------------


def _labels(text: str, patterns: tuple[_Pattern, ...] = _ALL_FILE_PATTERNS) -> set[str]:
    return {label for hit in _scan_text(text, patterns) for label in hit.labels}


_KNOWN_BAD: tuple[tuple[str, str], ...] = (
    # decision records
    ("(`decisions/0009`)", "decision-record path"),
    ("see decisions/0023 for the rule", "decision-record number"),
    ("``docs/decisions/0023-fit-diagnostics``", "decision-record path"),
    ("per `decisions/0008`, `0009`, `0011`", "decision-record number"),
    ("what `0011` forecloses", "decision-record number"),
    ("`0023` Amendment 2", "decision-record number"),
    ("0023 Amendment 2, item 4", "decision-record number"),
    ("decision 0011 says so", "decision-record number"),
    ("(0008, 0009)", "decision-record number"),
    ("0010.", "decision-record number"),
    ("see 0023", "decision-record number"),
    ("0031-0034", "decision-record number"),
    ("bare iD 0009-0005-3586-7901 without the URL", "decision-record number"),
    # section sign
    ("label §5.3-§5.5", "section sign"),
    ("(§3.1)", "section sign"),
    # the plan
    ("(plan section 5.4)", "plan"),
    ("the plan says so", "plan"),
    ("Plan §5.1", "plan"),
    ("loose floors (plan R9)", "plan"),
    ("the Phase 4 plan's answer", "plan"),
    ("the plans", "plan"),
    # private document names
    ("see IMPLEMENTATION_PLAN.md", "private document name"),
    ("per ROADMAP section 3.1", "private document name"),
    ("`REVIEWS.md` R8 #18", "private document name"),
    ("`CONSUMERS.md`", "private document name"),
    ("`ARCHITECTURE.md` A9", "private document name"),
    ("``STATISTICS.md``", "private document name"),
    ("docs/validation.md", "private document name"),
    ("CLAUDE.md hard constraint 6", "private document name"),
    ("PITCH.md", "private document name"),
    ("see ARCHITECTURE for why", "private document name"),
    # hard constraints
    ("(hard constraint 3)", "hard constraint"),
    ("Hard Constraint 2 forbids it", "hard constraint"),
    ("hard-constraint 1", "hard constraint"),
    # owner, decision-style: every tracked file
    ("pending owner decision", "owner decision (decision-style)"),
    ("Owner decision 2026-10-15", "owner decision (decision-style)"),
    ("the owner's decision", "owner decision (decision-style)"),
    ("owner decisions D1-D7", "owner decision (decision-style)"),
    ("an owner-approved exception", "owner decision (decision-style)"),
    ("Owner-Approved", "owner decision (decision-style)"),
    ("(owner, 2026-09-12)", "owner decision (decision-style)"),
    ("( owner approval)", "owner decision (decision-style)"),
    # decision labels
    ("pending D5", "decision label D1-D7"),
    ("(D7)", "decision label D1-D7"),
    ("D1, D2 and D3", "decision label D1-D7"),
    ("owner decision D5's restatement", "decision label D1-D7"),
    # phases
    ("Phase 4", "phase number"),
    ("(Phase 7)", "phase number"),
    ("phase 5 stats", "phase number"),
    ("in Phases 3 and 4", "phase number"),
    # appendix
    ("a recorded exception to Appendix B item 4", "Appendix B"),
    # requirement labels and review references
    ("(R2)", "requirement label R1-R10"),
    ("identical fits (R1): everything", "requirement label R1-R10"),
    ('R10 "Still open" item 1', "requirement label R1-R10"),
    ("`REVIEWS.md` R6 #14", "requirement label R1-R10"),
    ("R8 #18", "requirement label R1-R10"),
    ("(R3; R5)", "requirement label R1-R10"),
    # review-finding labels
    ("finding C2", "review-finding label"),
    ("(C1), so predict(0)", "review-finding label"),
    ("likelihood-path convergence (W3)", "review-finding label"),
    ("amendment, W1)", "review-finding label"),
    ("# S3, Phase 6 stats review", "review-finding label"),
    ("`ARCHITECTURE.md` A9", "review-finding label"),
    ("W2 / the observed information", "review-finding label"),
    # agent names
    ("which `api-compat` caught", "agent name"),
    ("written by test-author", "agent name"),
    ("a stats-reviewer finding", "agent name"),
    ("ask the implementer", "agent name"),
    # docs/ paths
    ("``docs/PROVENANCE.md``", "docs/ path"),
    ("see docs/STATISTICS.md", "docs/ path"),
    ("../docs/anything", "docs/ path"),
    ("https://github.com/robustmargin/robustmargin/blob/main/docs/ROADMAP.md", "docs/ path"),
    ("https://github.com/other/other/blob/main/docs/PROVENANCE.md", "docs/ path"),
    # dangling references into an unpublished document
    ('(decision\'s "Amendment 1")', "Amendment"),
    ("a later amendment", "Amendment"),
    ("(Amendments 2-4)", "Amendment"),
    ("(section 5.6)", "bare section number"),
    ("Section 7 of the document", "bare section number"),
    ("(round 4 sections 1-2)", "bare section number"),
    ("item 6", "bare item number"),
    ("Item 6: this call", "bare item number"),
    ("stats-review fix: an absolute tolerance", "stats review"),
    ("Stats review added the rule", "stats review"),
    ("the stats review", "stats review"),
    ("(design choice 2)", "design choice number"),
    ("Item 3, round 3: every object", "review round"),
    ("# Round 4, section 2: counts", "review round"),
    ("Round 2: which Fit.status", "review round"),
    ("rounds 3 and 4", "review round"),
    ("Rule 12: estimate is required", "numbered rule"),
    ("satisfying rule 12 (estimate must equal", "numbered rule"),
    ("Rules 1-3: a threshold's own status", "numbered rule"),
    # the retired names, case-insensitive, with or without a separator
    (f"the {_FORMER_PACKAGE} package", "retired package name"),
    (_FORMER_PACKAGE.upper(), "retired package name"),
    (_FORMER_PACKAGE.title(), "retired package name"),
    (f"{_FORMER_PACKAGE}_version", "retired package name"),
    ("margin-" + "kit", "retired package name"),
    ("margin_" + "kit", "retired package name"),
    ("margin " + "kit", "retired package name"),
    (f"src/{_FORMER_PACKAGE}/schema", "retired package name"),
    (f"matching {_SOURCE_REPO}'s convention", "retired source-repository name"),
    (f"../{_SOURCE_REPO}/results", "retired source-repository name"),
    (_SOURCE_REPO.upper(), "retired source-repository name"),
    (_SOURCE_REPO.title(), "retired source-repository name"),
    (_SOURCE_REPO.replace("-", "_"), "retired source-repository name"),
    (_SOURCE_REPO.replace("-", " "), "retired source-repository name"),
    (_SOURCE_REPO.replace("-", ""), "retired source-repository name"),
    (f"https://github.com/someone/{_SOURCE_REPO}", "retired source-repository name"),
    # the short form alone, in a file name or a sentence
    ("ze" + "ta_matrix_counts.csv", "retired source-repository name"),
    ("test_models_" + "ze" + "ta.py", "retired source-repository name"),
    ("the " + "Ze" + "ta curves", "retired source-repository name"),
    # the removed attribution file, uppercase
    ("see docs/PROVENANCE.md", "removed attribution file"),
    ("PROVENANCE.md", "removed attribution file"),
    ("# Provenance: PROVENANCE", "removed attribution file"),
    # pre-release version strings
    ("0.1.0a1", "pre-release 0.1.0aN"),
    ("v0.1.0a3", "pre-release 0.1.0aN"),
    ("introduced in 0.1.0a2.", "pre-release 0.1.0aN"),
    ("0.1.0", "release 0.1.0"),
    ("released as 0.1.0.", "release 0.1.0"),
    ("(0.1.0)", "release 0.1.0"),
    ("tag v0.1.0", "release 0.1.0"),
    ("version = '0.1.0'", "release 0.1.0"),
    ("0.1.1rc1", "pre-release 0.1.1rc1"),
    ("v0.1.1rc2 on TestPyPI", "pre-release 0.1.1rc1"),
    ("version 0.1.1rc1 on 2026-09-12", "pre-release 0.1.1rc1"),
    # a packaged-schema number other than 1
    # the retired series labels
    ("ppo_mass", "retired series label"),
    ("sac_wind", "retired series label"),
)


@pytest.mark.parametrize(
    ("text", "label"), _KNOWN_BAD, ids=[f"{label}: {t}" for t, label in _KNOWN_BAD]
)
def test_matcher_catches_known_bad_string(text: str, label: str) -> None:
    assert label in _labels(text), f"{text!r} should be caught as {label!r}; got {_labels(text)}"


_SHIPPED_KNOWN_BAD: tuple[str, ...] = (
    "the owner's answer",
    "which the owner chose",
    "pending owner decision",
    "Owners of the project",
    "owner",
)


@pytest.mark.parametrize("text", _SHIPPED_KNOWN_BAD)
def test_strict_owner_rule_catches_known_bad_string_in_a_shipped_file(text: str) -> None:
    assert "owner" in _labels(text, (*_ALL_FILE_PATTERNS, *_SHIPPED_ONLY_PATTERNS))


def test_a_shipped_path_gets_the_strict_owner_rule_and_a_workflow_does_not() -> None:
    assert _patterns_for("src/robustmargin/report.py") == (
        *_ALL_FILE_PATTERNS,
        *_SHIPPED_ONLY_PATTERNS,
    )
    assert _patterns_for("README.md") == (*_ALL_FILE_PATTERNS, *_SHIPPED_ONLY_PATTERNS)
    assert _patterns_for("pyproject.toml") == (*_ALL_FILE_PATTERNS, *_SHIPPED_ONLY_PATTERNS)
    assert _patterns_for("src/robustmargin/schema/scorecard-v1.json") == (
        *_ALL_FILE_PATTERNS,
        *_SHIPPED_ONLY_PATTERNS,
    )
    assert _patterns_for(".github/workflows/release.yml") == _ALL_FILE_PATTERNS
    assert _patterns_for(".github/CODEOWNERS") == _ALL_FILE_PATTERNS
    assert _patterns_for("tests/unit/test_report.py") == _ALL_FILE_PATTERNS
    assert _patterns_for("tools/check_dist.py") == _ALL_FILE_PATTERNS


# ------------------------------------------------------------------------------------------
# 4. Pinning the matcher: known-good strings pass.
# ------------------------------------------------------------------------------------------


_KNOWN_GOOD: tuple[str, ...] = (
    # The R language is a public reference, not a private document.
    "R `drc`",
    "R ``drc``'s ``logLik()``",
    'matches R ``drc``\'s ``type = "relative"``',
    "R's glm",
    "R's own ``binomial()$linkinv`` clamp",
    "a `glm` fit in R",
    "tools/drc_reference/README.md",
    # "planned", "planning" and kin are ordinary words.
    "planned for 0.2",
    "CI-width planning",
    "a planned release; planning a design; planar; explanation",
    "implemented, implementation, an implementable rule",
    # "owner" as an ordinary role: the code owner, PyPI's term for a project's GitHub organization.
    "This content is being reviewed by the code owner.",
    "# Code owner for the whole repository. GitHub requests this owner's review on every PR.",
    "# An email entry resolves only if it is a verified email on the owner's GitHub account.",
    "# environment requires the owner's approval, so nothing reaches PyPI without it.",
    "# One-time setup, done by the owner:",
    "owner `robustmargin` (the GitHub organization), repository `robustmargin`,",
    "ownership of the file handle",
    # The lowercase `provenance` is the public result field.
    "provenance: dict[str, object]",
    "an opaque ``provenance`` dict that the package stores and never interprets",
    "Scorecard(results=(), provenance={})",
    "payload['provenance']",
    # Version strings that are not the retired ones.
    "0.1.1",
    "version 0.1.1 on 2026-09-12",
    "0.1.10",
    "10.1.0",
    "1.0.1.0",
    "0.1.0.5",
    "numpy>=1.24.0, scipy>=1.10.0",
    "0.1.2rc1",
    "1.0.0a1",
    # Schema numbers: 1 is the only one, and a glob or template carries none.
    "scorecard-v1",
    "scorecard-v2.json",
    "axis_name='sensor_noise'",
    "scorecard-v1.json",
    "urn:robustmargin:schema:scorecard-v1",
    "schema_dir.glob('scorecard-v*.json')",
    "f'scorecard-v{version}.json'",
    # A public ORCID iD URL is an identifier, not a decision-record citation.
    "orcid: https://orcid.org/0009-0005-3586-7901",
    "https://orcid.org/0000-0002-1825-0097",
    # Numeric literals that contain four digits with a leading zero.
    "0.0025",
    "alpha = 0.0025",
    "1e-0025",
    "1.5E-0025",
    "x = -0.0025,",
    "10025",
    "1_0025",
    "0x0025",
    "0.00250",
    "tolerance 1e-10, ratio 0.0100",
    # Literature and method names.
    "Clopper-Pearson",
    "Albert & Anderson (1984)",
    "Fieller (1954), Wilson (1927), Dunn & Smyth (1996)",
    "Pearson dispersion, Newton's method, the Wald interval",
    "a decision rule on the p-value",
    "D-optimal design; Cook's D",
    "a phase-type distribution; out of phase; Phase of the moon",
    "this section describes the interface; each item of the list",
    "round the result; rounded to 3 places; a round trip through JSON; round-trip equality",
    "the rule is units-independent; a rule of thumb; the rules above; each rule holds",
    "an amended schema; the amended text",
    "Cartesian product A x B; S_3; H0 versus H1",
    "chemical C2H5OH and C1.5",
    "the pitch of the thread; the roadmap of a trip",
    # The neutral fixture labels.
    "system_a_axis_1",
    "series,axis,severity,successes,trials",
    "a consumer package named marginbench",
    "robustmargin.testing.fake_fit",
)


@pytest.mark.parametrize("text", _KNOWN_GOOD)
def test_matcher_passes_known_good_string(text: str) -> None:
    hits = _scan_text(text, _ALL_FILE_PATTERNS)

    assert not hits, f"{text!r} is legitimate public text but was flagged: {hits}"


def test_the_assembled_retired_names_are_the_names_this_guard_exists_to_forbid() -> None:
    # The fragments must reassemble into the two names; a typo in either would silently turn
    # the name patterns into patterns for something else.
    assert _FORMER_PACKAGE == "margin" + "kit"
    assert len(_FORMER_PACKAGE) == 9
    assert _SOURCE_REPO == "ze" + "ta" + "-bench"
    assert len(_SOURCE_REPO) == 10
    assert _labels(_FORMER_PACKAGE, _NAME_PATTERNS) == {"retired package name"}
    assert _labels(_SOURCE_REPO, _NAME_PATTERNS) == {"retired source-repository name"}


def test_every_pattern_has_a_known_bad_example_so_none_can_rot_into_matching_nothing() -> None:
    covered = {label for _text, label in _KNOWN_BAD}
    declared = {label for label, _ in (*_ALL_FILE_PATTERNS, *_SHIPPED_ONLY_PATTERNS)}

    # The strict `owner` rule is pinned by its own parametrised test above.
    assert declared - {"owner"} <= covered, (
        f"patterns with no known-bad example: {sorted(declared - {'owner'} - covered)}"
    )
    assert "owner" in declared
