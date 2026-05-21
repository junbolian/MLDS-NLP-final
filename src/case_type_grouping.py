"""Map raw Multi-LexSum `case_type` strings into 5 consolidated groups.

Multi-LexSum cases come tagged with case-type categories from the Civil
Rights Litigation Clearinghouse (CRLC). The raw label set in v20230518 has
24 distinct categories with significant long-tail imbalance, which would
hurt multi-class classification trained directly on raw labels.

We collapse them into 5 thematically coherent groups. The exact mapping
and rationale are in `docs/case_type_grouping.md`.

Label normalization: Multi-LexSum uses inconsistent whitespace around
`/` separators across labels (e.g. `"Public Benefits / Government Services"`
has spaces, `"Election/Voting Rights"` does not). The `_canonicalize`
function below normalizes whitespace around slashes so lookup is robust
to either form.
"""

from __future__ import annotations

import re
from typing import Optional

# Mapping from grouped category -> list of raw case_type strings.
# Labels below are the EXACT strings observed in v20230518 — verified by
# inspecting `df['case_type_raw'].value_counts()` on the full filtered set.
# Keep keys in their observed form (with or without spaces around `/`);
# the canonicalize step makes lookup tolerant either way.
CASE_TYPE_GROUPS: dict[str, list[str]] = {
    "Criminal Justice": [
        "Prison Conditions",
        "Jail Conditions",
        "Policing",
        "Juvenile Institution",
        "Criminal Justice (Other)",
        "Indigent Defense",
    ],
    "Civil Rights & Equality": [
        "Equal Employment",
        "Fair Housing/Lending/Insurance",
        "Public Accomm./Contracting",
        "School Desegregation",
        "Environmental Justice",
        "Public Housing",
    ],
    "Healthcare & Disability": [
        "Disability Rights-Pub. Accom.",
        "Mental Health (Facility)",
        "Intellectual Disability (Facility)",
        "Nursing Home Conditions",
    ],
    "Immigration & Education": [
        "Immigration and/or the Border",
        "Education",
        "Child Welfare",
    ],
    "Speech & Voting": [
        "Speech and Religious Freedom",
        "Election/Voting Rights",
        "Public Benefits / Government Services",
        "National Security",
        "Presidential/Gubernatorial Authority",
    ],
}


def _canonicalize(label: str) -> str:
    """Normalize whitespace around `/` so 'A/B', 'A / B', 'A /B' all match."""
    return re.sub(r"\s*/\s*", " / ", label.strip())


# Build a reverse lookup. Register BOTH the canonicalized form (spaces
# around /) and the no-space form, so direct lookup is O(1) for either.
_RAW_TO_GROUP: dict[str, str] = {}
for _group, _raws in CASE_TYPE_GROUPS.items():
    for _raw in _raws:
        canon = _canonicalize(_raw)
        _RAW_TO_GROUP[canon] = _group
        _RAW_TO_GROUP[canon.replace(" / ", "/")] = _group


def group_case_type(raw: Optional[str]) -> str:
    """Map a raw case_type to its grouped category.

    Unknown / missing labels are routed to "Other" — verify the EDA does
    not leak too many cases into that bucket. If "Other" > 5% of data,
    inspect `df['case_type_raw'].value_counts()` to find the unmapped
    labels and extend `CASE_TYPE_GROUPS` above.
    """
    if raw is None:
        return "Other"
    return _RAW_TO_GROUP.get(_canonicalize(raw), "Other")


def list_groups() -> list[str]:
    """Return the canonical group names (stable order)."""
    return list(CASE_TYPE_GROUPS.keys())


if __name__ == "__main__":
    # Smoke test against all 24 observed labels in v20230518
    observed_labels = [
        "Immigration and/or the Border", "Prison Conditions",
        "Public Benefits / Government Services", "Equal Employment",
        "Policing", "Jail Conditions", "Criminal Justice (Other)",
        "Speech and Religious Freedom", "National Security", "Education",
        "Election/Voting Rights", "Disability Rights-Pub. Accom.",
        "Presidential/Gubernatorial Authority", "Fair Housing/Lending/Insurance",
        "Child Welfare", "Juvenile Institution", "Mental Health (Facility)",
        "Public Accomm./Contracting", "Intellectual Disability (Facility)",
        "School Desegregation", "Indigent Defense", "Environmental Justice",
        "Public Housing", "Nursing Home Conditions",
    ]
    print(f"Testing {len(observed_labels)} observed raw labels:")
    n_other = 0
    for lab in observed_labels:
        g = group_case_type(lab)
        marker = "❌" if g == "Other" else "✓"
        print(f"  {marker} {lab!r:>50} -> {g!r}")
        if g == "Other":
            n_other += 1
    print(f"\nUnmapped: {n_other} / {len(observed_labels)}")
