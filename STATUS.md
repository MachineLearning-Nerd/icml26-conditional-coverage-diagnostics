# Status

Current step: exact external data provenance is recovered; validate deterministic
preparation and repair the released benchmark invocation before any run.

- Both author repositories are pinned and vendored in `upstream/`.
- The source contains a CPU path for the core eight-dataset LightGBM benchmark,
  its five-fold ERT evaluations, and ten-seed batch configuration.
- The source deliberately excludes `data/`; Appendix H's eight data sources are
  now pinned by URL, raw SHA-256 and expected shape in
  `repro/src/prepare_source_data.py`. This includes legacy OpenML `3050`
  (QSAR-TID-11 / `o11`) and `43093` (MiamiHousing2016), plus original Kaggle
  Food Delivery v1 rather than the mismatching newer TabArena curation.
- The submitted batch files call a non-existent
  `_generate_simultaneous_experiments_csv.py`. The only present classifier
  driver imports a missing local `ERT.py` and has most Table-2 methods
  commented out. See `docs/RELEASE_AUDIT.md`. These release defects must be
  disclosed; any runnable comparison will use an auditable external wrapper,
  not a silent edit to vendored source.
- TabPFN/TabICL are optional GPU-oriented comparators. They are never evidence
  for a CPU claim and will be reported separately if infeasible on allowed
  hardware.
- Nothing has been queued, published, or presented as a result.

Next action: run deterministic data preparation against the pinned raw files,
independently validate its manifest, then execute a small source smoke test.

Update 2026-07-28: deterministic preparation passed all eight Appendix-H
shape checks with raw and prepared hashes. `repro/jobs/source_smoke.py` is
ready to run the clean CPU import test and persist `source-smoke.json` to the
existing private jobs-artifacts dataset, but the Hub rejected submission before
compute (`429` shared API/repository-creation cooldown). No job ID, result, or
claim follows from that rejected submission. Retry only after the shared
cooldown has expired; continue local audit work meanwhile.
