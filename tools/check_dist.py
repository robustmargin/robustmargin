"""Check the built sdist and wheel in ``dist/`` before they are published.

Run after ``python -m build``: ``python tools/check_dist.py [--expect-version X.Y.Z]``. Both the
``packaging shape`` CI job and the release workflow call this file, so there is one copy of the
check.

A missing ``py.typed`` fails silently and disables type checking for every consumer, and
``load_schema(version)`` reads the packaged schemas at runtime, so both are asserted inside the
built wheel rather than merely on disk.

Both artifacts are also checked against an allowlist of what may ship: the package's modules,
``py.typed``, the score-card schema, ``LICENSE``, ``pyproject.toml``, the build's own metadata,
and ``README.md`` as the only documentation, since it is the PyPI project page. Tests, tools and
every other repository file never ship. The check is an allowlist rather than
a list of known-private paths, so a new file in an artifact fails it until someone decides it
belongs there.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import re
import tarfile
import zipfile

# Top-level modules only: a new subpackage fails this check until it is added deliberately.
_PACKAGE = r"robustmargin/(?:[A-Za-z_][A-Za-z0-9_]*\.py|py\.typed|schema/scorecard-v[0-9]+\.json)"
_WHEEL_ALLOWED = re.compile(
    rf"{_PACKAGE}"
    r"|robustmargin-[^/]+\.dist-info/(?:METADATA|WHEEL|RECORD|top_level\.txt|licenses/LICENSE)"
)
_SDIST_ALLOWED = re.compile(
    r"PKG-INFO|setup\.cfg|pyproject\.toml|README\.md|LICENSE"
    rf"|src/{_PACKAGE}"
    r"|src/robustmargin\.egg-info/"
    r"(?:PKG-INFO|SOURCES\.txt|dependency_links\.txt|requires\.txt|top_level\.txt)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        help="fail unless both artifacts carry exactly this version (the release tag, minus 'v')",
    )
    args = parser.parse_args()

    wheels = glob.glob("dist/*.whl")
    sdists = glob.glob("dist/*.tar.gz")
    assert len(wheels) == 1, f"expected exactly one wheel in dist/, found {wheels}"
    assert len(sdists) == 1, f"expected exactly one sdist in dist/, found {sdists}"
    wheel, sdist = wheels[0], sdists[0]

    names = zipfile.ZipFile(wheel).namelist()
    assert "robustmargin/py.typed" in names, f"py.typed missing from {wheel}"
    # Derived from the source tree rather than hardcoded, so a schema_version bump cannot leave
    # this check silently behind.
    expected_schemas = sorted(
        f"robustmargin/schema/{p.name}"
        for p in pathlib.Path("src/robustmargin/schema").glob("scorecard-v*.json")
    )
    assert expected_schemas, "no scorecard-v*.json found in src/robustmargin/schema"
    for schema in expected_schemas:
        assert schema in names, f"{schema} missing from {wheel}"

    unexpected_wheel = [n for n in names if not n.endswith("/") and not _WHEEL_ALLOWED.fullmatch(n)]
    assert not unexpected_wheel, f"files not on the allowlist in {wheel}: {unexpected_wheel}"

    # Members are prefixed by the sdist's top-level directory, `robustmargin-<version>/`.
    with tarfile.open(sdist) as tar:
        members = [
            m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile() and "/" in m.name
        ]
    assert "LICENSE" in members, f"LICENSE missing from {sdist}"
    assert "README.md" in members, f"README.md missing from {sdist}"
    unexpected_sdist = [n for n in members if not _SDIST_ALLOWED.fullmatch(n)]
    assert not unexpected_sdist, f"files not on the allowlist in {sdist}: {unexpected_sdist}"
    docs = sorted(
        n
        for n in members + names
        if n.lower().endswith((".md", ".rst", ".txt"))
        and "egg-info" not in n
        and "dist-info" not in n
    )
    assert docs == ["README.md"], f"README.md must be the only shipped document, found {docs}"

    if args.expect_version is not None:
        for artifact in (wheel, sdist):
            name = pathlib.Path(artifact).name
            assert f"robustmargin-{args.expect_version}" in name, (
                f"{name} does not carry version {args.expect_version}"
            )

    print(
        f"py.typed and {len(expected_schemas)} schema file(s) present in wheel "
        f"({', '.join(expected_schemas)}); LICENSE present; every file in both artifacts is on "
        f"the allowlist, and README.md is the only document shipped"
        + (f"; version {args.expect_version} confirmed" if args.expect_version else "")
    )


if __name__ == "__main__":
    main()
