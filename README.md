## Assumptions

- Rolling mean is computed using pandas rolling().
- The first (window - 1) rows produce NaN rolling means.
- Comparisons with NaN evaluate to False, therefore signal = 0 for those rows.
- Metrics JSON is written for both success and error scenarios.