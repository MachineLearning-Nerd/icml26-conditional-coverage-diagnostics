# Conclusion

---
<!-- trackio-cell
{"type":"markdown","id":"vaap_conclusion","created_at":"2026-08-02T22:01:33+00:00","title":"Conclusion","pinned":true,"pinned_at":"2026-08-02T22:01:33+00:00"}
-->
Claims 1, 3, and 4 retain their exact live-judged evidence. The new Claim 6 packet verifies the literal purpose of Algorithm 1: recorded test folds never overlap their training folds, and cross-fitting removes the false conditional-coverage violation created when the same classifier is fitted and scored in-sample.

Across five seeds, no-CV mean L1-ERT is `0.097280`, `0.097352`, and `0.040688` at 2k, 10k, and 50k rows. Every cross-fitted result for k=2,3,5,10 is below `0.01` in magnitude and at least five times smaller. An independent standard-library checker recomputes all 270 recorded summary fields and rejects three broken controls.

Claims 2 and 5 are not called successful. Their exact benchmark protocols exceed the authorized two-hour CPU limit and remain `BLOCKED — >2h CPU`.

## Reproduce

```bash
python3 repro/src/audit_claim6.py
```

The output is deterministic and matches [the published audit](https://huggingface.co/spaces/DineshAI/vaApZm6MKM/blob/main/outputs/claim6_independent_audit.json). Full-run provenance is the [completed CPU Job](https://huggingface.co/jobs/DineshAI/6a6c481323ed89c748ec92cd) at Git commit `c6f68ec340b9e01a261a02e753666721bf210645`; source and release notes are in the [public repository](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics).
