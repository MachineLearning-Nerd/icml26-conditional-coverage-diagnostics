# Executive summary

---
<!-- trackio-cell
{"type":"markdown","id":"vaap_exec_summary","created_at":"2026-08-02T22:01:31+00:00","title":"Executive summary","pinned":true,"pinned_at":"2026-08-02T22:01:31+00:00"}
-->
This CPU-only reproduction of [Conditional Coverage Diagnostics for Conformal Prediction](https://arxiv.org/abs/2512.11779) by Sacha Braun, David Holzmüller, Michael I. Jordan, and Francis Bach preserves the live-judged Claim 1, 3, and 4 evidence byte-for-byte. It adds a concise, independent audit for Claim 6: the recorded folds exactly match independent k-fold partitions, and cross-fitting removes a large spurious ERT violation for five seeds, three sample sizes, and every `k` in `{2,3,5,10}`.

Claims 2 and 5 remain explicitly blocked rather than overstated. Their literal protocols exceed the two-hour CPU ceiling: Claim 2 needs all seven methods in its normalization, and Claim 5's CIFAR10 predictor fits alone took 8,233–9,812 seconds before any ERT cell completed.

## Scope & cost

| Item | This reproduction | Literal paper scope |
| --- | --- | --- |
| Claims retained | 1, 3, 4 | Already live-judged 2/2 each |
| New candidate | Claim 6, exact k-fold and anti-overfitting audit | Algorithm 1 |
| Claim 6 scale | 5 seeds; 2k, 10k, 50k rows; k=2,3,5,10 | Finite-sample k-fold estimator |
| Controls | No-CV overfit arm; exact partition audit; 3 checker mutations | Avoid training/evaluation overlap |
| Hardware | Hugging Face `cpu-upgrade`, 64 vCPU | CPU-feasible |
| Runtime | 161.9 s full stage; <1 s independent audit | No GPU |
| Cost | Existing CPU Job; no new paid compute | USD 0 for this repair cycle |
| Blockers | Claims 2 and 5: `>2h CPU` | Full benchmark/image protocols |

Evidence: [Claim 6 raw result](https://huggingface.co/spaces/DineshAI/vaApZm6MKM/blob/main/raw/claim6_algorithm1__raw.json), [independent audit](https://huggingface.co/spaces/DineshAI/vaApZm6MKM/blob/main/outputs/claim6_independent_audit.json), [completed CPU Job](https://huggingface.co/jobs/DineshAI/6a6c481323ed89c748ec92cd), and [public GitHub repository](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics).

---
<!-- trackio-cell
{"type":"figure","id":"vaap_poster","created_at":"2026-08-02T22:01:32+00:00","title":"Reproduction poster (poster_embed.html)","pinned":true,"pinned_at":"2026-08-02T22:01:32+00:00","poster":true}
-->
<iframe src="poster_embed.html" title="Conditional coverage diagnostics reproduction poster" style="width:100%;height:680px;border:0"></iframe>
