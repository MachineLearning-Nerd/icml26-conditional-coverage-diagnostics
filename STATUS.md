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

Next action: package the repaired CPU runner with durable checkpointing, then
run the full eight-dataset ten-seed protocol on permitted `cpu-upgrade` once
the shared Hub cooldown permits job submission.

Update 2026-07-28: deterministic preparation passed all eight Appendix-H
shape checks with raw and prepared hashes. `repro/jobs/source_smoke.py` is
ready to run the clean CPU import test and persist `source-smoke.json` to the
existing private jobs-artifacts dataset, but the Hub rejected submission before
compute (`429` shared API/repository-creation cooldown). No job ID, result, or
claim follows from that rejected submission. Retry only after the shared
cooldown has expired; continue local audit work meanwhile.

Update 2026-07-28 (local CPU smoke): the `probmetrics` import required the
additional explicit `numba` dependency. The source-derived LightGBM and
PartitionWise paths then completed source ERT's five-fold control. The source
RealMLP → conformalizer path also completed on an 800-row real diamonds smoke
after forcing `device="cpu"`; leaving `device=None` attempted the incompatible
local GTX 1050 and failed with `cudaErrorNoKernelImageForDevice`. The observed
0.9125 smoke coverage is a setup control only, never a paper-scale claim.

Update 2026-07-28 (durable runner): `repro/src/run_full_repaired_cpu.py`
implements one complete Appendix-H dataset/seed at full data scale, including
the source 40/10/50 split, CPU RealMLP, ten test sizes, and five-fold ERT for
the source-derived CPU LightGBM and PartitionWise blocks. It atomically
checkpoints after each test size. `repro/jobs/full_cpu_entrypoint.py` runs it
from a read-only project mount with raw data and checkpoints on writable
`/data`. After cooldown, submit one job at a time with:

`hf jobs uv run --detach --flavor cpu-upgrade --timeout 8h --secrets HF_TOKEN -v .:/workspace:ro -v hf://buckets/DineshAI/jobs-artifacts:/data:rw repro/jobs/full_cpu_entrypoint.py <dataset> <seed>`

The local project was committed at `f88c53d`.

Update 2026-07-28 (full local CPU): Ailerons seed 0 completed the complete
paper-scale protocol and atomically wrote all ten test-size checkpoints to
`outputs/full-cpu/ailerons_seed0.json`. Its exact split sizes were
5,500/1,375/6,875; observed test coverage was 0.9173818. This is one raw seed
of ten and is not an aggregate, paper claim, queue candidate, or publication.

Update 2026-07-28 (full local CPU): Ailerons seed 1 completed all ten source
test sizes with the same 5,500/1,375/6,875 split and 0.9096727 raw test
coverage. Seeds 0 and 1 are complete raw outputs; no averaging, claim, queue
entry, or publication is authorized before seeds 2–9 and independent checks.

Update 2026-07-28 (full local CPU): Ailerons seed 2 completed all ten source
test sizes with 0.9070545 raw test coverage. Seeds 0–2 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 3 completed all ten source
test sizes with 0.8968727 raw test coverage. Seeds 0–3 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 4 completed all ten source
test sizes with 0.8855273 raw test coverage. Seeds 0–4 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 5 completed all ten source
test sizes with 0.8949818 raw test coverage. Seeds 0–5 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 6 completed all ten source
test sizes with 0.9000727 raw test coverage. Seeds 0–6 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 7 completed all ten source
test sizes with 0.9040000 raw test coverage. Seeds 0–7 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 8 completed all ten source
test sizes with 0.8805818 raw test coverage. Seeds 0–8 are complete raw
outputs; none is an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Ailerons seed 9 completed all ten source
test sizes with 0.8928000 raw test coverage. The complete Ailerons 10-seed
source-protocol unit is ready for strict aggregation and independent audit;
it is still only one of eight Appendix-H datasets and not a paper claim.

Current active computation: local CPU Diamonds seed 0 uses the pinned 53,940
row Appendix-H source data and the same idempotent full-scale runner.

Update 2026-07-28 (full local CPU): Diamonds seed 0 completed all ten source
test sizes with 0.9069707 raw test coverage. It is one raw seed of ten and is
not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 1 completed all ten source
test sizes with 0.8975899 raw test coverage. Its full 26,970-example coverage
record passed the persisted count, covered-total, and SHA-256 integrity checks.
It is a second raw seed of ten and is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 2 completed all ten source
test sizes with 0.8982944 raw test coverage. Its full 26,970-example coverage
record passed the persisted count, covered-total, and SHA-256 integrity checks.
It is a third raw seed of ten and is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 3 completed all ten source
test sizes with 0.8966259 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is a fourth raw seed of ten
and is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 4 completed all ten source
test sizes with 0.8979607 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is a fifth raw seed of ten and
is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 5 completed all ten source
test sizes with 0.9081201 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is a sixth raw seed of ten and
is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 6 completed all ten source
test sizes with 0.9019651 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is a seventh raw seed of ten
and is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 7 completed all ten source
test sizes with 0.8926956 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is an eighth raw seed of ten
and is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 8 completed all ten source
test sizes with 0.8999258 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. It is a ninth raw seed of ten and
is not an aggregate or paper claim.

Update 2026-07-28 (full local CPU): Diamonds seed 9 completed all ten source
test sizes with 0.9038932 raw test coverage. Its full 26,970-example coverage
record passed the independent checkpoint audit, including persisted count,
covered-total, and SHA-256 integrity checks. The ten raw seeds have a strict
one-dataset aggregate of 0.9004042 mean coverage (SEM 0.0015258). This is not
a paper claim; its legacy seed 0 integrity repair remains queued below.

Update 2026-07-28 (full local CPU): WineQuality completed all ten source seeds
at its full 6,497-row scale. Its strict aggregate mean coverage is 0.8942154
(SEM 0.0032929), and an independent required-integrity audit passed all ten
full 3,250-example coverage records. This is one Appendix-H dataset, not a
paper claim.

Update 2026-07-28 (full local CPU): Miami2016 completed all ten source seeds
at its full 13,932-row scale. Its strict aggregate mean coverage is 0.8998134
(SEM 0.0025028), and an independent required-integrity audit passed all ten
full 6,967-example coverage records. This is one Appendix-H dataset, not a
paper claim.

Update 2026-07-28 (full local CPU): O11 completed all ten source seeds at its
full 5,742-row scale. Its strict aggregate mean coverage is 0.9056058 (SEM
0.0033696), and an independent required-integrity audit passed all ten full
2,872-example coverage records. This is one Appendix-H dataset, not a paper
claim.

Current active computation: the fail-closed master queue is running local CPU
Superconductivity seed 7, then will run seeds 8–9 and independently audit and
strictly aggregate it before serially advancing through the two remaining
Appendix-H datasets. It does not publish or create a claim.

Queue integrity hardening: the legacy Ailerons seeds 0–9 and Diamonds seed 0
have the correct protocol and ten-size data but predate persisted coverage
digest recording. A fourth detached, non-overlapping repair waits for the
six-dataset Appendix-H queue to finish, preserves those legacy checkpoints,
reruns them with the current source-faithful runner, and then requires
integrity checks and strict aggregation on all eight datasets before writing a
single Appendix-H integrity manifest. This is evidence hardening only and does
not create a claim.

Update 2026-07-28 (queue hardening): the full-scale runner now resumes an
incomplete atomic result instead of treating its existence as completion. It
recomputes and verifies the pinned split/model coverage contract first, retains
only matching saved test-size metrics, and consumes the original seeded sample
draws before skipping retained sizes. This preserves the source sampling order
on recovery and makes the serial queues fail closed on incompatible artifacts.

Update 2026-07-28 (deterministic foundation evidence): an independent verifier
now calls the pinned `covmetrics` implementation on an exact 90%-coverage
construction and separately records the source estimator's fitted/tested
partitions. The constant 0.9 predictor has zero L1/L2/KL excess risk, whereas
an intentionally nonconstant candidate cannot improve it; 95% and 85%
constructions isolate the positive over and under L1 components respectively;
and all five source KFold partitions exactly match an independent KFold audit.
This is evidence for anchored claims 1, 4, and 6 only. The full-scale
eight-dataset and synthetic claims remain required before publication.

Queued next computation: a detached zero-CPU synthetic queue waits for the
all-eight Appendix-H integrity manifest, then runs ten serial atomic synthetic
seeds using the released 300,000-test-point generator and all fifteen released
test sizes. It evaluates the pinned L1-ERT field directly and records the
known conditional-coverage truth; it does not relabel the legacy driver's
Brier-only field as L1-ERT.
