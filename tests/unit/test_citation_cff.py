"""``CITATION.cff`` carries the metadata Zenodo and GitHub read, and is never shipped.

Zenodo archives each GitHub Release and takes the record's metadata from ``CITATION.cff``; GitHub
uses it for "Cite this repository". The file is tracked but stays out of the wheel and the sdist.
It deliberately has no ``version`` and no ``date-released``: the version lives only in the package
``__init__``, and Zenodo takes the version and date from the release tag.

PyYAML is not a dependency, so the few top-level keys needed are read with line logic.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CFF = _ROOT / "CITATION.cff"

_ORCID_URL = "https://orcid.org/0009-0005-3586-7901"


def _text() -> str:
    assert _CFF.is_file(), f"{_CFF} does not exist"
    return _CFF.read_text(encoding="utf-8")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _top_level_keys(text: str) -> list[str]:
    return re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", text, flags=re.MULTILINE)


def _scalar(text: str, key: str) -> str:
    """The value of a top-level ``key: value`` line (single-line scalars only)."""
    matches = re.findall(rf"^{re.escape(key)}:[ \t]*(.*)$", text, flags=re.MULTILINE)
    assert len(matches) == 1, f"expected exactly one top-level {key!r} line, found {len(matches)}"
    return _unquote(matches[0])


def _block(text: str, key: str) -> list[str]:
    """The indented lines under a top-level ``key:`` (up to the next top-level key)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"{re.escape(key)}:", ln)), None)
    assert start is not None, f"no top-level {key!r} key"
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if ln.strip() and not ln.startswith((" ", "\t", "-")):
            break
        body.append(ln)
    return body


def _orcid_checksum_char(base_digits: str) -> str:
    """ISO 7064 MOD 11-2 check character for the first 15 digits of an ORCID iD."""
    total = 0
    for ch in base_digits:
        total = (total + int(ch)) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)


def test_citation_file_declares_cff_version_type_message_and_title() -> None:
    text = _text()

    assert _scalar(text, "cff-version") == "1.2.0"
    assert _scalar(text, "type") == "software"
    assert _scalar(text, "message") != ""
    assert _scalar(text, "title") == "robustmargin"


def test_citation_license_is_apache_and_equals_the_pyproject_license() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project_license = tomllib.load(handle)["project"]["license"]

    assert _scalar(_text(), "license") == "Apache-2.0"
    assert _scalar(_text(), "license") == project_license


def test_citation_repository_code_is_the_public_repository() -> None:
    assert _scalar(_text(), "repository-code") == "https://github.com/robustmargin/robustmargin"


def test_citation_has_exactly_one_author_with_the_expected_identity() -> None:
    body = _block(_text(), "authors")
    entries = [ln for ln in body if re.match(r"\s*-\s", ln)]
    assert len(entries) == 1, f"expected exactly one author entry, found {len(entries)}"

    fields: dict[str, str] = {}
    for ln in body:
        match = re.match(r"\s*(?:-\s+)?([a-z-]+):[ \t]*(.*)$", ln)
        if match:
            fields[match.group(1)] = _unquote(match.group(2))

    assert fields == {
        "family-names": "Badkoubeh",
        "given-names": "Babak",
        "orcid": _ORCID_URL,
        "affiliation": "Independent Researcher",
    }


def test_orcid_checksum_is_valid_iso_7064_mod_11_2() -> None:
    orcid = _scalar_in_authors_orcid()
    digits = orcid.removeprefix("https://orcid.org/").replace("-", "")

    assert re.fullmatch(r"\d{15}[\dX]", digits), f"malformed ORCID iD {orcid!r}"
    assert _orcid_checksum_char(digits[:15]) == digits[15]


def _scalar_in_authors_orcid() -> str:
    for ln in _block(_text(), "authors"):
        match = re.match(r"\s*(?:-\s+)?orcid:[ \t]*(.*)$", ln)
        if match:
            return _unquote(match.group(1))
    pytest.fail("no orcid in authors")


def test_orcid_checksum_routine_accepts_the_published_example_and_rejects_a_typo() -> None:
    # ORCID's documentation example https://orcid.org/0000-0002-1825-0097 is valid;
    # changing a digit makes it invalid.
    assert _orcid_checksum_char("000000021825009") == "7"
    assert _orcid_checksum_char("000000021825008") != "7"


def test_citation_has_nonempty_abstract_and_keywords() -> None:
    text = _text()
    assert _scalar_or_block_nonempty(text, "abstract")
    assert _scalar_or_block_nonempty(text, "keywords")
    assert any(re.match(r"\s*-\s+\S", ln) for ln in _block(text, "keywords"))


def _scalar_or_block_nonempty(text: str, key: str) -> bool:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        match = re.match(rf"{re.escape(key)}:[ \t]*(.*)$", ln)
        if match:
            inline = match.group(1).strip()
            if inline and inline not in (">", ">-", "|", "|-", ">+", "|+"):
                return True
            return any(nxt.strip() for nxt in _block(text, key))
    return False


@pytest.mark.parametrize("key", ["version", "date-released"])
def test_citation_omits_version_and_release_date_which_come_from_the_release_tag(
    key: str,
) -> None:
    assert key not in _top_level_keys(_text())


def _check_dist_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dist", _ROOT / "tools" / "check_dist.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_allowlist_rejects_citation_file_in_both_artifacts() -> None:
    module = _check_dist_module()
    wheel_allowed = module.__dict__["_WHEEL_ALLOWED"]
    sdist_allowed = module.__dict__["_SDIST_ALLOWED"]
    assert isinstance(wheel_allowed, re.Pattern)
    assert isinstance(sdist_allowed, re.Pattern)

    # The allowlist patterns are live: the files that do ship are admitted.
    assert sdist_allowed.fullmatch("LICENSE")
    assert wheel_allowed.fullmatch("robustmargin/py.typed")

    # The sdist members are checked with the top-level directory stripped; the wheel's are
    # root-relative. Neither admits the citation file, at the root or inside the package.
    for name in ("CITATION.cff", "robustmargin/CITATION.cff", "src/robustmargin/CITATION.cff"):
        assert not sdist_allowed.fullmatch(name), f"sdist allowlist admits {name}"
        assert not wheel_allowed.fullmatch(name), f"wheel allowlist admits {name}"
