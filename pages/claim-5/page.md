# Claim 5 - Table-4 classification decomposition

---
<!-- trackio-cell
{"type":"markdown","id":"claim5_honest_blocker","created_at":"2026-08-03T00:00:02+00:00","title":"Claim 5 status and blocker","pinned":true,"pinned_at":"2026-08-03T00:00:02+00:00"}
-->
## BLOCKED — >2h CPU

> Classification experiments report divergent KL+-ERT and KL--ERT values across conformal prediction methods, demonstrating the over/under-coverage decomposition in practice.

Source: [Table 4 and Section 4.3.2](https://arxiv.org/html/2512.11779v1#S4.T4).

This claim is neither verified nor falsified. Its literal scope is four datasets (MNIST, FashionMNIST, CIFAR10, CIFAR100), two conformal strategies, ten repeats, and five-fold ERT. Three source-faithful dataset jobs trained every predictor, but none completed an ERT cell before termination; therefore no partial predictor result is relabeled as claim evidence.

| Dataset | Exact attempt | Observed outcome |
| --- | --- | --- |
| MNIST | [Job `6a6c56ae…`](https://huggingface.co/jobs/DineshAI/6a6c56ae23ed89c748ec941f) | Ten predictors completed; no ERT cell before exit 143 |
| FashionMNIST | [Job `6a6c5905…`](https://huggingface.co/jobs/DineshAI/6a6c5905b36a6516e96a39c7) | Ten predictors completed; no ERT cell before exit 143 |
| CIFAR10 | [Job `6a6c590e…`](https://huggingface.co/jobs/DineshAI/6a6c590e23ed89c748ec946f) | Predictor fits took 8,233–9,812 s, already above two hours |
| CIFAR100 | Not launched | Released driver trains ResNet-18 for 35 epochs inside every ERT fold |

The attempted code is retained in the [MNIST](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics/tree/audit/c5-mnist), [FashionMNIST](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics/tree/audit/c5-fashionmnist), and [CIFAR10](https://github.com/MachineLearning-Nerd/icml26-conditional-coverage-diagnostics/tree/audit/c5-cifar10) branches. The official experiment code remains pinned to `ElSacho/Conditional_Coverage_Estimation@39a99dcad92205a15d93f2c5fec40c76540abf1c` and the metric implementation to `ElSacho/covmetrics@a5205aada6a0f39e3812daf087753217ef66b159`.

Unblock condition: complete all eight dataset/strategy cells and their ten repetitions inside the allowed compute envelope. Until then the honest score is 0/2.
