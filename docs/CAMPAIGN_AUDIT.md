# Campaign audit

## Scope fixed before repair

The six official claims come from the 2026-08-02 fetch of `claims_anchored.json` for OpenReview `vaApZm6MKM`. Claim anchors were audited against arXiv HTML `2512.11779v1`, SHA-256 `1e13e76905dab485726ab80b671ca7140c73e58cdc6001832379f4a349da5a03`. The official repositories are pinned to:

- `ElSacho/Conditional_Coverage_Estimation@39a99dcad92205a15d93f2c5fec40c76540abf1c`
- `ElSacho/covmetrics@a5205aada6a0f39e3812daf087753217ef66b159`

Claims 1, 3, and 4 already had live 2/2 verdicts and were treated as read-only. Their published page SHA-256 values remain:

- Claim 1: `7dbe488df829cbe000543ca7133da24e66cad89b63a5ef432e29e80ee97185fd`
- Claim 3: `bed4be85bf0a9be4fa00553252a599010684bb2ff197a6ac36c456cf1c2ed4fd`
- Claim 4: `30592a20e122095a42e8f3a690c51d6d3b29b5bd9a6119aa5882a7c92faf5245`

## Weak-claim triage

### Claim 2

The literal Table-2 percentage is normalized over all seven methods, all sizes, four datasets, and ten repeats. A reduced method set changes the denominator and cannot reproduce 68.4% versus 38.3%. Measured foundation-model costs were 2.6 CPU-hours per experiment on the smallest dataset and 5.1–5.9 hours on the larger datasets; the full matrix is about 55 CPU-box-hours. Decision: `BLOCKED — >2h CPU`.

### Claim 5

The literal Table-4 scope is four datasets, two strategies, ten repeats, and five-fold ERT. Source-faithful CPU Jobs trained predictors but produced no ERT cell. CIFAR10 predictor fits alone took 8,233–9,812 seconds, and CIFAR100 adds ResNet-18 training inside every fold. Decision: `BLOCKED — >2h CPU`.

Attempted Jobs:

- MNIST: `https://huggingface.co/jobs/DineshAI/6a6c56ae23ed89c748ec941f`
- FashionMNIST: `https://huggingface.co/jobs/DineshAI/6a6c5905b36a6516e96a39c7`
- CIFAR10: `https://huggingface.co/jobs/DineshAI/6a6c590e23ed89c748ec946f`

### Claim 6

The advance criteria were fixed before inspecting the summary: recorded partitions must equal an independent seeded KFold split; no-CV L1-ERT must exceed 0.02 at every size; and cross-fitted L1-ERT must be below 0.01 in magnitude and at least five times smaller for k=2,3,5,10 at 2k, 10k, and 50k rows.

The completed CPU Job `https://huggingface.co/jobs/DineshAI/6a6c481323ed89c748ec92cd` produced the full five-seed raw evidence at Git commit `c6f68ec340b9e01a261a02e753666721bf210645`. The stage completed in 161.9 seconds wall time and 133.4 seconds CPU time.

`repro/src/audit_claim6.py` is dependency-free and separate from the generating pipeline. It recomputes 270 stored statistics from per-seed values, checks all partitions and thresholds, and rejects three deliberately broken controls. Two clean runs produced the same output bytes:

- raw SHA-256: `59aeb547948d67117da86f28d3708c572ba296b45594bc3c656f5307e1843852`
- audit SHA-256: `2e08d3f03901cd97db882598cbaa2c70dcbf5aa62e930833af8d29ca9127483b`

Decision: `VERIFIED` by the blind packet. Only the live judge can bank the points.

## No-regression rule

The judged revision is `214cfb6aabee9c072106bb80bc2b888f356442b8`. Its 28 files and 12 page nodes remain present. The release allowlist only adds or replaces UTF-8 text paths; no binary, LFS, Xet, Trackio binary-write, or force-push path is used.
