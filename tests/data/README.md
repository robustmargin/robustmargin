# `pooled_counts.csv`

Binomial counts pooled per severity level, for five series. The tests read this file and assert
fixed expectations on it: fit statuses, censoring labels, break points and, for the one
identifiable series, numeric threshold values. The counts are therefore fixed literals the tests
rely on. A changed count is a different dataset, not a fixture update.

## Columns

`series, axis, severity, successes, trials`

- `series` -- one curve, named `{system}_{axis}` (for example `system_a_axis_1`).
- `axis` -- the severity axis the curve runs along: `axis_1`, `axis_2` or `axis_3`.
- `severity` -- the level on that axis. `0.0`, where present, is the unperturbed control level.
- `successes`, `trials` -- the counts pooled over every run at that level: a sum of counts, not
  an average of rates, so `successes / trials` is the pooled rate at that level.

`axis_3` is signed (`-0.2` to `+0.2`); every other axis is unsigned.

## Series and the data shape each exercises

| Series | Axis | Shape |
|---|---|---|
| `system_a_axis_1` | `axis_1` | Identifiable transition: success falls from 200/200 at the control to 0/300 at the highest level |
| `system_a_axis_2` | `axis_2` | No failure anywhere: 100% success at every level, so the fit is completely separated |
| `system_b_axis_1` | `axis_1` | The control is already below the gate (98/200), and no success is observed at any nonzero level |
| `system_c_axis_1` | `axis_1` | Non-monotone: the rate rises from the control (57/200) to the first nonzero level (143/300), then collapses to 0/300; the control is also below the gate |
| `system_c_axis_3` | `axis_3` | Signed and asymmetric: the negative side declines more slowly than the positive side |
