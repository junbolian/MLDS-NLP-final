# `case_type` Grouping Decision

Multi-LexSum v20230518 exposes **24 raw case-type labels** from the Civil Rights Litigation Clearinghouse (CRLC). Training a classifier on all 24 would suffer from long-tail imbalance (the top 5 labels cover ~60% of cases; several have fewer than 10 examples).

We collapse the 24 raw labels into **5 thematically coherent groups**. After the final mapping, **0 cases fall into "Other"** — all observed labels are explicitly placed.

## Final mapping (all 24 labels)

| Group | Original CRLC Labels | Cases | Rationale |
|-------|---------------------|-------|-----------|
| **Criminal Justice** (577) | `Prison Conditions`; `Jail Conditions`; `Policing`; `Juvenile Institution`; `Criminal Justice (Other)`; `Indigent Defense` | 577 | All concern state/local custodial or law-enforcement contexts. Indigent Defense fits here (criminal-court representation for poor defendants). |
| **Speech & Voting** (414) | `Speech and Religious Freedom`; `Election/Voting Rights`; `Public Benefits / Government Services`; `National Security`; `Presidential/Gubernatorial Authority` | 414 | First Amendment + civic-participation cases, plus executive-power constitutional balancing (Presidential/Gubernatorial Authority typically involves separation-of-powers challenges to EOs, gubernatorial orders, etc.). |
| **Immigration & Education** (363) | `Immigration and/or the Border`; `Education`; `Child Welfare` | 363 | Cases concerning recent arrivals and minors; share agency-review + due-process + substantive-rights-for-vulnerable-populations frames. |
| **Civil Rights & Equality** (179) | `Equal Employment`; `Fair Housing/Lending/Insurance`; `Public Accomm./Contracting`; `School Desegregation`; `Environmental Justice`; `Public Housing` | 179 | Anti-discrimination cases. School Desegregation fits squarely under Brown v. Board / 14th Amendment doctrine. Environmental Justice and Public Housing belong here because they're equity-frame cases (disparate impact on protected groups). |
| **Healthcare & Disability** (69) | `Disability Rights-Pub. Accom.`; `Mental Health (Facility)`; `Intellectual Disability (Facility)`; `Nursing Home Conditions` | 69 | Cases turning on health status, capacity, and access to care. Disability Rights-Pub. Accom. sits here despite the "Public Accommodations" suffix because the substantive law (ADA) is disability-centered. |

## Why 5 groups, not more or fewer?

- 5 satisfies Option 2's "more than 2 case types" requirement and demonstrates non-trivial multi-class learning.
- 5 is small enough that each group has enough training examples (smallest is 69, largest is 577) for sklearn classifiers + Bi-LSTM. Use `class_weight='balanced'` to compensate for the imbalance ratio (~8:1 worst case).
- All groups have substantive thematic overlap in vocabulary and legal doctrine, so the groups should be learnable from text features.

## Label normalization

Multi-LexSum uses inconsistent whitespace around `/` separators (e.g. `"Public Benefits / Government Services"` has spaces, `"Election/Voting Rights"` does not). The function `group_case_type()` in `src/case_type_grouping.py` calls `_canonicalize()` to insert single spaces around all slashes before lookup, so the mapping is robust to either form.

## How to add a new mapping

If a future Multi-LexSum version introduces a new raw label:

1. Re-run `python -m src.cleaning --force`; the log will print `'Other' group is X.X%`. If > 0%, the new label is unmapped.
2. Dump the unmapped labels:
   ```python
   df = pd.read_parquet('data/multilexsum_clean.parquet')
   print(df.loc[df['case_type_grouped']=='Other', 'case_type_raw'].value_counts())
   ```
3. Add the new label string (in its observed form) to the most appropriate group's list in `src/case_type_grouping.py`.
4. Re-run cleaning + EDA.
5. Note the addition in the table above with a one-line rationale.
