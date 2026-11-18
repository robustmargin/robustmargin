"""Serialisation between robustmargin's result dataclasses and JSON: :func:`to_dict`,
:func:`from_dict`, :class:`Scorecard`, and :func:`load_schema`.

``to_dict`` walks a result object and produces a plain, ``json.dumps(..., allow_nan=False)``-safe
structure: enums become their string values, tuples become lists, and :class:`Scorecard`,
:class:`~robustmargin.GridBreakPoint`, :class:`~robustmargin.Fit`, :class:`~robustmargin.Threshold`
and :class:`~robustmargin.Ratio` are tagged with a ``"type"`` key naming the class, at every
nesting depth (a ``Fit`` embedded in a ``Threshold`` is tagged too). The five helper types
(:class:`~robustmargin.Axis`, :class:`~robustmargin.Definition`, :class:`~robustmargin.Cell`,
:class:`~robustmargin.Parameter`, :class:`~robustmargin.Covariance`) are never tagged. A
non-finite float anywhere raises ``ValueError``, and a ``provenance`` value outside plain JSON (a
tuple or a set, for example) raises ``TypeError`` -- no NaN and nothing un-JSON-encodable ever
reaches ``json.dumps``.

``from_dict`` is the inverse: it dispatches on ``"type"``, rebuilds nested objects, and raises
``ValueError`` naming the problem. At each level, ``schema_version`` (when the type has one) is
checked before an unknown field is: an unrecognised ``schema_version`` is reported on its own
even if the same dict also carries an unknown field. An unknown field found anywhere inside a
``Scorecard`` -- at the ``Scorecard``'s own level or at any nesting depth beneath it -- names
the field, the enclosing card's ``library_version``, and this reader's version; a bare
``Fit``/``Threshold``/``Ratio``/``GridBreakPoint`` read on its own (not nested in a
``Scorecard``) gets a plainer message with no version context, since there is no card to name.
A key that a tagged type's schema ``required`` list requires but the dict omits raises; a key
the schema does not require, if omitted, is filled from the dataclass's own default. This
required-key set is read from the packaged schema (:data:`load_schema`), not duplicated as a
literal list, so it cannot drift from ``schema/scorecard-v3.json``.

``SCHEMA_VERSION`` is ``"3"``, which is what every result carries by default and what
``to_dict`` writes for a freshly built result, and ``from_dict`` reads ``"1"``, ``"2"`` and
``"3"``. A ``schema_version`` of anything else, on any typed object, raises a ``ValueError`` that
names the version found and the versions this reader reads. Version ``"2"`` adds the
``"increasing"`` direction and the ``"baseline_multiple"`` definition to version ``"1"``; a
``Fit`` or ``Threshold`` tagged ``"1"`` that carries either raises, since no ``"1"`` writer could
have produced it (the check is per object, not per card). The same holds for a ``Fit`` with
``model="gaussian"`` (a continuous response), which schema ``"2"`` also carries: its ``cells`` are
``ContinuousCell``, it has a ``residual``, and its diagnostics are ``ContinuousDiagnostics``. A
binomial ``Fit`` writes no ``residual`` key at all, so its serialised form is unchanged.

Version ``"3"`` adds the isotonic model (``model="isotonic"``, ``link="none"``), the
``"parametric_bootstrap"``, ``"cell_bootstrap"`` and ``"cluster_bootstrap"`` interval methods,
``Fit.cell_estimates``, ``Threshold.bootstrap`` and, for clustered and paired inference,
``Fit.dependence`` (``"clustered"``), ``Fit.cluster_cells``, the cluster sign test of
``MonotonicityCheck``, ``Threshold.dependence`` (``"clustered"``), ``Ratio.dependence``
(``"paired"``), ``Ratio.method`` (``"cluster_bootstrap"``) and ``Ratio.bootstrap``. ``to_dict``
writes the keys ``cell_estimates``, ``bootstrap`` and ``cluster_cells``, as ``null`` when unset,
and a ``Fit``'s ``dependence``, except on a result tagged ``"1"`` or ``"2"``, where an unset one
(and an ``"independent"`` ``Fit.dependence``) is left out. A result tagged ``"1"`` or ``"2"`` that
carries any of the new values raises, since no earlier writer could have produced it; a card from
an earlier release (no such keys, or explicit ``null`` or ``"independent"`` ones) still reads.
A result read from a ``"1"`` or ``"2"`` card keeps that ``schema_version``, as it did before
version ``"3"``, so writing it back gives a card the release that wrote it can still read. Only
the ``Scorecard`` root requires the tag; a nested or bare result that omits it is read as ``"3"``.
A reader older than this release rejects every card written now, so upgrade readers before
writers.

``load_schema`` reads the packaged ``schema/scorecard-v{version}.json`` with
:mod:`importlib.resources`; ``scorecard-v1.json``, ``scorecard-v2.json`` and
``scorecard-v3.json`` are packaged.
Nothing under ``src/`` imports ``jsonschema``; it stays a dev-only dependency used by
consumers' own validators and by this package's own tests.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

from robustmargin.empirical import BaselineRate, GridBreakPoint
from robustmargin.models import Covariance, Fit, Parameter
from robustmargin.ratio import Ratio
from robustmargin.threshold import Threshold
from robustmargin.types import (
    Axis,
    BootstrapSummary,
    Cell,
    Censoring,
    ClusterAdjacentPair,
    ClusterCell,
    ContinuousCell,
    Definition,
    IntervalShape,
    JSONValue,
    Residual,
    Status,
)
from robustmargin.validation import (
    AdjacentPair,
    ContinuousAdjacentPair,
    ContinuousDiagnostics,
    Diagnostics,
    MonotonicityCheck,
)

__all__ = ["SCHEMA_VERSION", "Scorecard", "from_dict", "load_schema", "to_dict"]

SCHEMA_VERSION = "3"

# `from_dict` reads schemas "1", "2" and "3".
_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2", "3"})

# What schema "2" added to "1": a "1" payload carrying either is not one a "1" writer produced.
_VERSION_2_DIRECTIONS = frozenset({"increasing"})
_VERSION_2_DEFINITION_KINDS = frozenset({"baseline_multiple"})
_VERSION_2_MODELS = frozenset({"gaussian"})
# What schema "3" added to "2": a "1" or "2" payload carrying any of it is not one those writers
# produced.
_VERSION_3_MODELS = frozenset({"isotonic"})
_VERSION_3_LINKS = frozenset({"none"})
_VERSION_3_INTERVAL_METHODS = frozenset(
    {"parametric_bootstrap", "cell_bootstrap", "cluster_bootstrap"}
)
_VERSION_3_KEYS = ("cell_estimates", "bootstrap", "cluster_cells")
# What a "1" or "2" writer never wrote for ``dependence`` (on a Fit, Threshold or Ratio), for a
# Ratio's ``method`` and for a Fit's nested monotonicity check.
_VERSION_3_DEPENDENCE = frozenset({"clustered", "paired"})
_VERSION_3_RATIO_METHODS = frozenset({"cluster_bootstrap"})
_CLUSTER_SIGN_TEST_METHOD = "cluster_sign_test_one_sided_holm"
# A field written only when set: a binomial Fit has no residual and writes no key for it, so its
# serialised form is the one a schema "1" reader accepts.
_OMITTED_WHEN_NONE = frozenset({"residual"})


def _default_created_at() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _default_library_version() -> str:
    # Imported lazily, not at module level, to avoid a circular import between this module and
    # robustmargin/__init__.py (which imports result types report.py also imports).
    import robustmargin

    return robustmargin.__version__


@dataclass(frozen=True, kw_only=True)
class Scorecard:
    """A collection of results, plus the metadata that lets a saved file be read back years
    later: which robustmargin version produced it, when, and under what schema.

    Attributes
    ----------
    results
        A tuple of results, each a :class:`~robustmargin.GridBreakPoint`,
        :class:`~robustmargin.Fit`, :class:`~robustmargin.Threshold` or
        :class:`~robustmargin.Ratio`. Any other element type raises ``TypeError``.
    provenance
        An opaque mapping the caller may attach to record where the run came from. robustmargin
        stores it and never interprets it. Empty by default.
    created_at
        An ISO-8601 datetime string carrying a timezone. Defaults to now, in UTC. A given value
        that :func:`datetime.datetime.fromisoformat` cannot parse, or that carries no timezone,
        raises ``ValueError``.
    library_version
        The version of robustmargin that produced this scorecard. Defaults to
        :data:`robustmargin.__version__` at construction time.
    schema_version
        The serialised-result schema version this object belongs to. Defaults to ``"3"``, the
        version ``to_dict`` writes; ``from_dict`` also reads ``"1"`` and ``"2"``.
    """

    results: tuple[GridBreakPoint | Fit | Threshold | Ratio, ...]
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)
    created_at: str = field(default_factory=_default_created_at)
    library_version: str = field(default_factory=_default_library_version)
    schema_version: str = "3"

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        for item in self.results:
            if not isinstance(item, (GridBreakPoint, Fit, Threshold, Ratio)):
                raise TypeError(
                    "Scorecard.results entries must each be a GridBreakPoint, Fit, Threshold "
                    f"or Ratio, got {type(item).__name__}"
                )
        try:
            parsed = dt.datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError(
                f"Scorecard.created_at must be an ISO-8601 datetime string, got {self.created_at!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError(f"Scorecard.created_at must carry a timezone, got {self.created_at!r}")


_TAGGED_CLASSES: tuple[type, ...] = (Scorecard, GridBreakPoint, Fit, Threshold, Ratio)


# --------------------------------------------------------------------------------------------
# to_dict
# --------------------------------------------------------------------------------------------


def _encode_json_value(value: object) -> JSONValue:
    """Encode a value that must already be plain JSON (a ``provenance`` entry): only
    ``None``/``bool``/``int``/``float``/``str``/``list``/``dict`` are accepted. A tuple, a set,
    or any other type raises ``TypeError``; a non-finite float raises ``ValueError``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in provenance: {value!r}")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_encode_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode_json_value(v) for k, v in value.items()}
    raise TypeError(
        "provenance values must be JSON (dict, list, str, int, float, bool or None), got "
        f"{type(value).__name__}: {value!r}"
    )


def _encode_provenance(value: Mapping[str, object]) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"provenance must be a mapping, got {type(value).__name__}")
    return {str(k): _encode_json_value(v) for k, v in value.items()}


def _encode_field_value(value: object) -> JSONValue:
    if isinstance(value, (Censoring, Status, IntervalShape)):
        return value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in result: {value!r}")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_dataclass_instance(value)
    if isinstance(value, (tuple, list)):
        return [_encode_field_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _encode_field_value(v) for k, v in value.items()}
    raise TypeError(f"cannot encode value of type {type(value).__name__}: {value!r}")


def _encode_dataclass_instance(obj: Any) -> dict[str, JSONValue]:
    cls = type(obj)
    result: dict[str, JSONValue] = {}
    for f in dataclasses.fields(obj):
        raw = getattr(obj, f.name)
        if f.name in _OMITTED_WHEN_NONE and raw is None:
            continue
        # A result tagged "1" or "2" is written as that version's writer wrote it, without the
        # keys that only "3" has, so the release that wrote the card can read it back.
        if (
            f.name in _VERSION_3_KEYS
            and raw is None
            and getattr(obj, "schema_version", None) in ("1", "2")
        ):
            continue
        # `dependence` is a required key of a Threshold, a Ratio and a MonotonicityCheck in every
        # version, so only a Fit leaves it out: a Fit written by an earlier release has no such
        # key.
        if (
            isinstance(obj, Fit)
            and f.name == "dependence"
            and raw == "independent"
            and obj.schema_version in ("1", "2")
        ):
            continue
        if f.name == "provenance":
            result[f.name] = _encode_provenance(raw)
        else:
            result[f.name] = _encode_field_value(raw)
    if cls in _TAGGED_CLASSES:
        result["type"] = cls.__name__
    return result


def to_dict(obj: object) -> dict[str, JSONValue]:
    """Encode a robustmargin result dataclass instance into a plain, JSON-safe dict.

    Parameters
    ----------
    obj
        A :class:`Scorecard`, :class:`~robustmargin.GridBreakPoint`, :class:`~robustmargin.Fit`,
        :class:`~robustmargin.Threshold` or :class:`~robustmargin.Ratio` instance -- the five tagged
        types. Nothing else is accepted at the top level (a helper type such as
        :class:`~robustmargin.Axis` only ever appears nested inside one of these).

    Returns
    -------
    dict[str, JSONValue]
        A dict safe to pass to ``json.dumps(..., allow_nan=False)``. Tagged types carry a
        ``"type"`` key naming the class, at every nesting depth.

    Raises
    ------
    ValueError
        If any float anywhere in ``obj`` (including inside ``provenance``) is NaN or infinite.
    TypeError
        If ``obj`` is not one of the five tagged types, or if a ``provenance`` value is not
        plain JSON.
    """
    if not isinstance(obj, _TAGGED_CLASSES):
        raise TypeError(
            "to_dict expects a Scorecard, GridBreakPoint, Fit, Threshold or Ratio instance, "
            f"got {type(obj).__name__}"
        )
    return _encode_dataclass_instance(obj)


# --------------------------------------------------------------------------------------------
# from_dict
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _schema_required_by_type() -> dict[str, frozenset[str]]:
    """The packaged schema's own ``required`` list for each of the five tagged result types
    (:class:`Scorecard`, :class:`~robustmargin.GridBreakPoint`, :class:`~robustmargin.Fit`,
    :class:`~robustmargin.Threshold`, :class:`~robustmargin.Ratio`), keyed by class name, with
    ``"type"`` removed -- ``from_dict`` checks ``"type"`` itself, separately.

    Read from :func:`load_schema` rather than duplicated as a literal list, so a required field
    can never drift between the schema and ``from_dict`` without the schema/``from_dict`` field
    parity test (or this function's own malformed-schema check) catching it.
    """
    schema = load_schema()
    root_required = schema.get("required")
    if not isinstance(root_required, list):
        raise TypeError(
            f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing root "
            "'required'"
        )
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        raise TypeError(
            f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing '$defs'"
        )

    result: dict[str, frozenset[str]] = {
        "Scorecard": frozenset(str(k) for k in root_required if k != "type")
    }
    for name in ("GridBreakPoint", "Fit", "Threshold", "Ratio"):
        node = defs.get(name)
        if not isinstance(node, dict):
            raise TypeError(
                f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing "
                f"'$defs.{name}'"
            )
        node_required = node.get("required")
        if not isinstance(node_required, list):
            raise TypeError(
                f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing "
                f"'$defs.{name}.required'"
            )
        result[name] = frozenset(str(k) for k in node_required if k != "type")
    return result


@lru_cache(maxsize=1)
def _schema_properties_by_type() -> dict[str, frozenset[str]]:
    """The packaged schema's own ``properties`` key set for each of the five tagged result
    types, keyed by class name, ``"type"`` **included** (unlike
    :func:`_schema_required_by_type`, which strips it since ``from_dict`` checks ``"type"``
    separately -- there is no such special-casing needed here).

    Used as :func:`_validated_fields`'s "allowed keys" set for a tagged type, in place of the
    dataclass's own field names, so a key the schema does not list is rejected as unknown even if
    a dataclass field of that name were ever added without the schema following. Same caching and
    malformed-schema-detection shape as :func:`_schema_required_by_type`, read from
    :func:`load_schema` rather than duplicated as a literal list.
    """
    schema = load_schema()
    root_properties = schema.get("properties")
    if not isinstance(root_properties, dict):
        raise TypeError(
            f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing root "
            "'properties'"
        )
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        raise TypeError(
            f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing '$defs'"
        )

    result: dict[str, frozenset[str]] = {"Scorecard": frozenset(str(k) for k in root_properties)}
    for name in ("GridBreakPoint", "Fit", "Threshold", "Ratio"):
        node = defs.get(name)
        if not isinstance(node, dict):
            raise TypeError(
                f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing "
                f"'$defs.{name}'"
            )
        node_properties = node.get("properties")
        if not isinstance(node_properties, dict):
            raise TypeError(
                f"robustmargin's packaged schema v{SCHEMA_VERSION} is malformed: missing "
                f"'$defs.{name}.properties'"
            )
        result[name] = frozenset(str(k) for k in node_properties)
    return result


def _validated_fields(
    cls: type, data: object, *, tagged: bool, card_version: str | None = None
) -> dict[str, Any]:
    """Check ``data``'s keys against ``cls``'s dataclass fields (and ``schema_version``, if
    ``cls`` has one), and return the subset of ``data`` corresponding to fields that are
    present.

    ``schema_version`` is checked before the unknown-field check, so an unsupported version is
    reported on its own even when the dict also carries an unknown key. A version outside
    :data:`_SUPPORTED_SCHEMA_VERSIONS` is reported as unknown. ``card_version`` is the
    ``library_version`` of the enclosing :class:`Scorecard`, threaded down from
    :func:`from_dict` for every object nested inside one; when it is not ``None``, an
    unknown-field error names it (and this reader's version) alongside the field. A bare
    ``Fit``/``Threshold``/``Ratio``/``GridBreakPoint`` read on its own (``card_version=None``)
    gets a plainer message, since there is no card to name.

    For one of the five tagged result types, a field is required exactly when the packaged
    schema says so (:func:`_schema_required_by_type`); for an untagged helper type
    (:class:`~robustmargin.Axis`, :class:`~robustmargin.Definition`,
    :class:`~robustmargin.Cell`, :class:`~robustmargin.Parameter`,
    :class:`~robustmargin.Covariance`), a field is required exactly when its dataclass field has
    no default. A required field missing from ``data`` raises ``ValueError``; every other missing
    field is simply omitted here, so the constructor applies its own default.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"from_dict: expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )

    fields = dataclasses.fields(cls)
    field_names = {f.name for f in fields}

    if "schema_version" in field_names:
        schema_version = data.get("schema_version", SCHEMA_VERSION)
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"from_dict: unknown schema_version {schema_version!r} for {cls.__name__}, "
                f"expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)!r}"
            )

    # A tagged type's allowed keys come from the packaged schema rather than from the dataclass's
    # field names, so the schema stays the single statement of what a card may contain. An
    # untagged helper type has no schema entry of its own to check against, so it allows exactly
    # its dataclass fields.
    allowed = _schema_properties_by_type()[cls.__name__] if tagged else field_names

    unknown = set(data) - allowed
    if unknown:
        if card_version is not None:
            reader_version = _default_library_version()
            raise ValueError(
                f"from_dict: unknown field(s) {sorted(unknown)} for {cls.__name__} (card "
                f"written by robustmargin {card_version}; this reader is robustmargin "
                f"{reader_version})"
            )
        raise ValueError(f"from_dict: unknown field(s) {sorted(unknown)} for {cls.__name__}")

    required = (
        _schema_required_by_type()[cls.__name__]
        if tagged
        else frozenset(
            f.name
            for f in fields
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        )
    )

    result: dict[str, Any] = {}
    for f in fields:
        if f.name in data:
            result[f.name] = data[f.name]
        elif f.name in required:
            raise ValueError(f"from_dict: missing required field {f.name!r} for {cls.__name__}")
        else:
            continue
    return result


def _expect_tag(data: object, *, expected: str) -> object:
    """Check that a nested tagged value carries ``"type": expected`` before it is rebuilt.

    A wrong or missing ``"type"`` on a nested value (for example a ``Threshold`` payload where
    ``Ratio.threshold_a`` expects a ``Threshold``) is a corrupted or hand-edited file, not
    something to coerce silently.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"from_dict: expected a mapping tagged type={expected!r}, got {type(data).__name__}"
        )
    tag = data.get("type")
    if tag != expected:
        raise ValueError(f"from_dict: expected type={expected!r}, got {tag!r}")
    return data


def _axis_from_dict(data: object, card_version: str | None) -> Axis:
    return Axis(**_validated_fields(Axis, data, tagged=False, card_version=card_version))


def _definition_from_dict(data: object, card_version: str | None) -> Definition:
    fields = _validated_fields(Definition, data, tagged=False, card_version=card_version)
    return Definition(**fields)


def _cell_from_dict(data: object, card_version: str | None) -> Cell:
    return Cell(**_validated_fields(Cell, data, tagged=False, card_version=card_version))


def _continuous_cell_from_dict(data: object, card_version: str | None) -> ContinuousCell:
    return ContinuousCell(
        **_validated_fields(ContinuousCell, data, tagged=False, card_version=card_version)
    )


def _residual_from_dict(data: object, card_version: str | None) -> Residual:
    return Residual(**_validated_fields(Residual, data, tagged=False, card_version=card_version))


def _parameter_from_dict(data: object, card_version: str | None) -> Parameter:
    return Parameter(**_validated_fields(Parameter, data, tagged=False, card_version=card_version))


def _covariance_from_dict(data: object, card_version: str | None) -> Covariance:
    fields = _validated_fields(Covariance, data, tagged=False, card_version=card_version)
    return Covariance(**fields)


def _adjacent_pair_from_dict(data: object, card_version: str | None) -> AdjacentPair:
    """``AdjacentPair`` is an untagged helper, like ``Cell`` and ``Parameter``: it only ever
    appears inside ``MonotonicityCheck.pairs``, never as a card's top-level result.
    """
    fields = _validated_fields(AdjacentPair, data, tagged=False, card_version=card_version)
    return AdjacentPair(**fields)


def _continuous_adjacent_pair_from_dict(
    data: object, card_version: str | None
) -> ContinuousAdjacentPair:
    fields = _validated_fields(
        ContinuousAdjacentPair, data, tagged=False, card_version=card_version
    )
    return ContinuousAdjacentPair(**fields)


def _cluster_adjacent_pair_from_dict(data: object, card_version: str | None) -> ClusterAdjacentPair:
    """``ClusterAdjacentPair`` is an untagged helper, like ``AdjacentPair``."""
    fields = _validated_fields(ClusterAdjacentPair, data, tagged=False, card_version=card_version)
    return ClusterAdjacentPair(**fields)


def _cluster_cell_from_dict(data: object, card_version: str | None) -> ClusterCell:
    return ClusterCell(
        **_validated_fields(ClusterCell, data, tagged=False, card_version=card_version)
    )


def _monotonicity_check_from_dict(data: object, card_version: str | None) -> MonotonicityCheck:
    """``MonotonicityCheck`` is an untagged helper, like ``BaselineRate``: it only ever appears
    as ``Fit.monotonicity``. Its ``method`` says which pair type its ``pairs`` are.
    """
    fields = _validated_fields(MonotonicityCheck, data, tagged=False, card_version=card_version)
    build_pair: Any = {
        "welch_t_one_sided_holm": _continuous_adjacent_pair_from_dict,
        _CLUSTER_SIGN_TEST_METHOD: _cluster_adjacent_pair_from_dict,
    }.get(str(fields.get("method")), _adjacent_pair_from_dict)
    fields["pairs"] = tuple(build_pair(p, card_version) for p in fields["pairs"])
    return MonotonicityCheck(**fields)


def _diagnostics_from_dict(data: object, card_version: str | None) -> Diagnostics:
    """``Diagnostics`` is an untagged helper, like ``BaselineRate``: it only ever appears as
    ``Fit.diagnostics``.
    """
    fields = _validated_fields(Diagnostics, data, tagged=False, card_version=card_version)
    fields["link_status"] = {k: Status(v) for k, v in fields["link_status"].items()}
    return Diagnostics(**fields)


def _continuous_diagnostics_from_dict(
    data: object, card_version: str | None
) -> ContinuousDiagnostics:
    """``ContinuousDiagnostics`` is an untagged helper, like ``Diagnostics``: it only ever appears
    as a Gaussian ``Fit.diagnostics``."""
    fields = _validated_fields(ContinuousDiagnostics, data, tagged=False, card_version=card_version)
    fields["link_status"] = {k: Status(v) for k, v in fields["link_status"].items()}
    return ContinuousDiagnostics(**fields)


def _reject_version_2_content(
    fields: dict[str, Any], *, type_name: str, definition: object = None, raw: object = None
) -> None:
    """Refuse a payload that says it is schema ``"1"`` but carries what only ``"2"`` has."""
    if fields.get("schema_version", SCHEMA_VERSION) != "1":
        return
    if fields.get("model") in _VERSION_2_MODELS:
        raise ValueError(
            f"from_dict: {type_name} has schema_version '1' but model {fields['model']!r}, "
            "which only schema_version '2' has"
        )
    if type_name == "Fit" and isinstance(raw, Mapping) and "residual" in raw:
        raise ValueError(
            "from_dict: Fit has schema_version '1' but carries a 'residual' key, which only "
            "schema_version '2' has"
        )
    if fields.get("direction") in _VERSION_2_DIRECTIONS:
        raise ValueError(
            f"from_dict: {type_name} has schema_version '1' but direction "
            f"{fields['direction']!r}, which only schema_version '2' has"
        )
    if isinstance(definition, Mapping) and definition.get("kind") in _VERSION_2_DEFINITION_KINDS:
        raise ValueError(
            f"from_dict: {type_name} has schema_version '1' but definition kind "
            f"{definition['kind']!r}, which only schema_version '2' has"
        )


def _reject_version_3_content(fields: dict[str, Any], *, type_name: str) -> None:
    """Refuse a payload that says it is schema ``"1"`` or ``"2"`` but carries what only ``"3"``
    has: a non-null ``cell_estimates``, ``bootstrap`` or ``cluster_cells``, the isotonic model, the
    ``"none"`` link, a bootstrap interval method, a clustered or paired ``dependence``, a ratio's
    cluster bootstrap method, or a Fit's nested cluster sign test. An explicit ``null`` for a new
    key, and an explicit ``"independent"`` ``dependence``, are what a version ``"3"`` writer emits
    for an unset field, and are read."""
    version = fields.get("schema_version", SCHEMA_VERSION)
    if version not in ("1", "2"):
        return
    for key in _VERSION_3_KEYS:
        if fields.get(key) is not None:
            raise ValueError(
                f"from_dict: {type_name} has schema_version {version!r} but carries a non-null "
                f"{key!r}, which only schema_version '3' has"
            )
    for key, only_in_3 in (
        ("model", _VERSION_3_MODELS),
        ("link", _VERSION_3_LINKS),
        ("interval_method", _VERSION_3_INTERVAL_METHODS),
        ("dependence", _VERSION_3_DEPENDENCE),
    ):
        if fields.get(key) in only_in_3:
            raise ValueError(
                f"from_dict: {type_name} has schema_version {version!r} but {key} "
                f"{fields[key]!r}, which only schema_version '3' has"
            )
    if type_name == "Ratio" and fields.get("method") in _VERSION_3_RATIO_METHODS:
        raise ValueError(
            f"from_dict: Ratio has schema_version {version!r} but method {fields['method']!r}, "
            "which only schema_version '3' has"
        )
    monotonicity = fields.get("monotonicity")
    if type_name == "Fit" and isinstance(monotonicity, Mapping):
        if (
            monotonicity.get("dependence") == "clustered"
            or monotonicity.get("method") == _CLUSTER_SIGN_TEST_METHOD
        ):
            raise ValueError(
                f"from_dict: Fit has schema_version {version!r} but carries a cluster sign test "
                "in its monotonicity check, which only schema_version '3' has"
            )


def _bootstrap_summary_from_dict(data: object, card_version: str | None) -> BootstrapSummary:
    """``BootstrapSummary`` is an untagged helper: it only appears as ``Threshold.bootstrap`` and
    ``Ratio.bootstrap``."""
    return BootstrapSummary(
        **_validated_fields(BootstrapSummary, data, tagged=False, card_version=card_version)
    )


def _fit_from_dict(data: object, card_version: str | None) -> Fit:
    fields = _validated_fields(Fit, data, tagged=True, card_version=card_version)
    _reject_version_2_content(fields, type_name="Fit", raw=data)
    _reject_version_3_content(fields, type_name="Fit")
    fields["axis"] = _axis_from_dict(fields["axis"], card_version)
    gaussian = fields.get("model") == "gaussian"
    continuous_cells = gaussian or (
        fields.get("model") == "isotonic"
        and bool(fields["cells"])
        and isinstance(fields["cells"][0], Mapping)
        and "n" in fields["cells"][0]
    )
    build_cell = _continuous_cell_from_dict if continuous_cells else _cell_from_dict
    fields["cells"] = tuple(build_cell(c, card_version) for c in fields["cells"])
    fields["status"] = Status(fields["status"])
    if fields.get("params") is not None:
        fields["params"] = {
            k: _parameter_from_dict(v, card_version) for k, v in fields["params"].items()
        }
    if fields.get("covariance") is not None:
        fields["covariance"] = _covariance_from_dict(fields["covariance"], card_version)
    if fields.get("cluster_ids") is not None:
        fields["cluster_ids"] = tuple(fields["cluster_ids"])
    if fields.get("cell_estimates") is not None:
        fields["cell_estimates"] = tuple(fields["cell_estimates"])
    if fields.get("cluster_cells") is not None:
        fields["cluster_cells"] = tuple(
            _cluster_cell_from_dict(c, card_version) for c in fields["cluster_cells"]
        )
    # Neither `diagnostics` nor `monotonicity` is in the schema's `required` list for a Fit, and
    # `_validated_fields` fills a key a card omits from the dataclass default (`None`), so a card
    # that omits either decodes with `None` and needs no special case beyond the nested-object
    # conversion below.
    if fields.get("diagnostics") is not None:
        build_diagnostics = (
            _continuous_diagnostics_from_dict if gaussian else _diagnostics_from_dict
        )
        fields["diagnostics"] = build_diagnostics(fields["diagnostics"], card_version)
    if fields.get("residual") is not None:
        fields["residual"] = _residual_from_dict(fields["residual"], card_version)
    if fields.get("monotonicity") is not None:
        fields["monotonicity"] = _monotonicity_check_from_dict(fields["monotonicity"], card_version)
    return Fit(**fields)


def _baseline_rate_from_dict(data: object, card_version: str | None) -> BaselineRate:
    """``BaselineRate`` is an untagged helper, like ``Cell`` and ``Parameter``: it only ever
    appears as ``Threshold.baseline``, never as a card's top-level result, so it carries no
    ``"type"`` discriminator.
    """
    fields = _validated_fields(BaselineRate, data, tagged=False, card_version=card_version)
    return BaselineRate(**fields)


def _threshold_from_dict(data: object, card_version: str | None) -> Threshold:
    fields = _validated_fields(Threshold, data, tagged=True, card_version=card_version)
    _reject_version_2_content(fields, type_name="Threshold", definition=fields.get("definition"))
    _reject_version_3_content(fields, type_name="Threshold")
    fields["axis"] = _axis_from_dict(fields["axis"], card_version)
    fields["definition"] = _definition_from_dict(fields["definition"], card_version)
    fields["censoring"] = Censoring(fields["censoring"])
    fields["status"] = Status(fields["status"])
    fields["fit"] = _fit_from_dict(_expect_tag(fields["fit"], expected="Fit"), card_version)
    # Neither `baseline` nor `grid` is in the schema's `required` list for a Threshold, and
    # `_validated_fields` fills a key a card omits from the dataclass default (`None`), so a card
    # that omits either decodes with `None` and needs no special case here.
    if fields.get("baseline") is not None:
        fields["baseline"] = _baseline_rate_from_dict(fields["baseline"], card_version)
    if fields.get("grid") is not None:
        fields["grid"] = _grid_break_point_from_dict(
            _expect_tag(fields["grid"], expected="GridBreakPoint"), card_version
        )
    if fields.get("bootstrap") is not None:
        fields["bootstrap"] = _bootstrap_summary_from_dict(fields["bootstrap"], card_version)
    return Threshold(**fields)


def _ratio_from_dict(data: object, card_version: str | None) -> Ratio:
    fields = _validated_fields(Ratio, data, tagged=True, card_version=card_version)
    _reject_version_3_content(fields, type_name="Ratio")
    fields["threshold_a"] = _threshold_from_dict(
        _expect_tag(fields["threshold_a"], expected="Threshold"), card_version
    )
    fields["threshold_b"] = _threshold_from_dict(
        _expect_tag(fields["threshold_b"], expected="Threshold"), card_version
    )
    if fields.get("shape") is not None:
        fields["shape"] = IntervalShape(fields["shape"])
    fields["censoring"] = Censoring(fields["censoring"])
    fields["status"] = Status(fields["status"])
    if fields.get("bootstrap") is not None:
        fields["bootstrap"] = _bootstrap_summary_from_dict(fields["bootstrap"], card_version)
    return Ratio(**fields)


def _grid_break_point_from_dict(data: object, card_version: str | None) -> GridBreakPoint:
    fields = _validated_fields(GridBreakPoint, data, tagged=True, card_version=card_version)
    fields["censoring"] = Censoring(fields["censoring"])
    return GridBreakPoint(**fields)


def _scorecard_from_dict(data: object, card_version: str | None) -> Scorecard:
    # ``card_version`` (uniform with every other builder in _RESULT_BUILDERS) is unused here: a
    # Scorecard is always the root of its own tree, never nested inside another Scorecard, so
    # its own library_version is what every unknown-field error beneath it names instead.
    if not isinstance(data, Mapping):
        raise ValueError(f"from_dict: expected a mapping for Scorecard, got {type(data).__name__}")
    own_version = data.get("library_version")
    own_version_str = own_version if isinstance(own_version, str) else "<unknown>"
    fields = _validated_fields(Scorecard, data, tagged=True, card_version=own_version_str)
    fields["results"] = tuple(
        _from_dict_impl(item, card_version=own_version_str) for item in fields["results"]
    )
    return Scorecard(**fields)


_RESULT_BUILDERS: dict[str, Any] = {
    "GridBreakPoint": _grid_break_point_from_dict,
    "Fit": _fit_from_dict,
    "Threshold": _threshold_from_dict,
    "Ratio": _ratio_from_dict,
    "Scorecard": _scorecard_from_dict,
}


def _from_dict_impl(data: object, *, card_version: str | None) -> object:
    if not isinstance(data, Mapping):
        raise ValueError(f"from_dict expects a mapping, got {type(data).__name__}")
    type_name = data.get("type")
    if type_name is None:
        raise ValueError("from_dict: missing required field 'type'")
    builder = _RESULT_BUILDERS.get(type_name)
    if builder is None:
        raise ValueError(f"from_dict: unknown type {type_name!r}")
    result: object = builder(data, card_version)
    return result


def from_dict(data: object) -> object:
    """Rebuild a robustmargin result object from the dict :func:`to_dict` produced (or an
    equivalent one read back through ``json.loads``).

    Parameters
    ----------
    data
        A mapping carrying a ``"type"`` key naming one of ``Scorecard``, ``GridBreakPoint``,
        ``Fit``, ``Threshold`` or ``Ratio``, plus that type's fields.

    Returns
    -------
    object
        An instance of the type named by ``data["type"]``. ``from_dict(to_dict(x)) == x`` for
        every ``x`` of these five types whose ``schema_version`` is ``"3"``, which is every
        result this package builds. ``schema_version`` ``"1"`` and ``"2"`` are read too, and the
        result they give keeps its ``schema_version``. A ``Fit`` or ``Threshold`` tagged ``"1"``
        carries no increasing outcome, no ``baseline_multiple`` definition and no Gaussian model,
        and one tagged ``"1"`` or ``"2"`` carries no isotonic model, bootstrap interval method,
        ``cell_estimates`` or ``bootstrap``, no clustered or paired ``dependence``, no
        ``cluster_cells``, no cluster sign test and no ``"cluster_bootstrap"`` ratio method; one
        that does is rejected (checked per object, not per card). Constructors do not check the
        tag, so an object built with any other ``schema_version`` serialises but does not read
        back.

    Raises
    ------
    ValueError
        If ``"type"`` is missing or unrecognised; if ``schema_version`` (at any nesting level)
        is not ``"1"``, ``"2"`` or ``"3"``, or is ``"1"`` on a ``Fit`` or ``Threshold`` carrying
        what only ``"2"`` has (an ``"increasing"`` direction, a ``"baseline_multiple"``
        definition or a Gaussian model), or ``"1"`` or ``"2"`` on one carrying what only ``"3"``
        has (the isotonic model, a bootstrap interval method, a clustered or paired
        ``dependence``, the cluster sign test, or a non-null ``cell_estimates``, ``bootstrap`` or
        ``cluster_cells``); if a field (at any nesting level) is not recognised; or if a required
        field is missing.
    """
    return _from_dict_impl(data, card_version=None)


# --------------------------------------------------------------------------------------------
# load_schema
# --------------------------------------------------------------------------------------------


def load_schema(version: str = SCHEMA_VERSION) -> dict[str, object]:
    """Load a packaged score-card JSON Schema (draft 2020-12).

    Reads ``schema/scorecard-v{version}.json`` from inside the installed ``robustmargin`` package
    using :mod:`importlib.resources`, so it works the same way from a wheel as from a source
    checkout. ``jsonschema`` itself is never imported here or anywhere under ``src/``; it is a
    dev-only dependency for consumers that want to validate against the schema.

    Parameters
    ----------
    version
        Which packaged schema version to load. Defaults to :data:`SCHEMA_VERSION`, ``"3"``.
        ``"1"`` and ``"2"`` are packaged too, for reading cards written by earlier releases.

    Returns
    -------
    dict[str, object]
        The parsed JSON Schema document.

    Raises
    ------
    FileNotFoundError
        If ``version`` names a schema that is not packaged. Only ``"1"``, ``"2"`` and ``"3"`` are.
    """
    text = (
        resources.files("robustmargin")
        .joinpath("schema", f"scorecard-v{version}.json")
        .read_text(encoding="utf-8")
    )
    schema: dict[str, object] = json.loads(text)
    return schema
