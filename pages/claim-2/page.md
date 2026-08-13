# Claim 2 - Table-2 relative power

---
<!-- trackio-cell
{"type":"markdown","id":"claim2_honest_blocker","created_at":"2026-08-03T00:00:01+00:00","title":"Claim 2 status and blocker","pinned":true,"pinned_at":"2026-08-03T00:00:01+00:00"}
-->
## BLOCKED — >2h CPU

> Using LightGBM as the underlying classifier for L1-ERT achieves 68.4% relative statistical power, compared with 38.3% for PartitionWise.

Source: [Table 2 and Section 4.1](https://arxiv.org/html/2512.11779v1#S4.T2).

This claim is neither verified nor falsified. The literal statistic normalizes over all seven methods, all test sizes, four datasets, and ten repeats. Removing TabPFN or TabICL changes the denominator and produces a different, systematically larger statistic.

The authors' committed CSV reproduces their reported values within 0.95 percentage points, which confirms the recovered aggregation formula but is not independent evidence. Full reruns measured the two CPU foundation-model routes at 2.6 hours per experiment on the smallest dataset and 5.1–5.9 hours on the larger datasets. The complete protocol is approximately 55 CPU-box-hours, exceeding the campaign's two-hour limit.

| Gate | Status |
| --- | --- |
| Exact aggregation formula recovered | PASS |
| All seven methods required | CONFIRMED |
| Independent four-dataset, ten-repeat result | MISSING |
| Faithful completion within two CPU hours | FAIL |

Source and attempted-run history are retained in the [public repository](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics), including the per-dataset Claim-2 branches. A five-method CPU subset is deliberately not presented as the paper's number.

Unblock condition: a faithful all-seven-method execution that completes within the permitted compute envelope. Until then the honest score is 0/2.
