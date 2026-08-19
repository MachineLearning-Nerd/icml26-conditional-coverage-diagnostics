# Primary-source and release audit

The detailed source record is [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md).
This top-level file is the short, stable pointer used by the repository's
publication contract.

## Paper identity

- Title: *Conditional Coverage Diagnostics for Conformal Prediction*
- Authors: Sacha Braun, David Holzmüller, Michael I. Jordan, and Francis Bach
- arXiv: [2512.11779](https://arxiv.org/abs/2512.11779)
- ICML submission: `vaApZm6MKM`
- Source URL audited: `https://arxiv.org/html/2512.11779v1`
- Source HTML SHA-256: `1e13e76905dab485726ab80b671ca7140c73e58cdc6001832379f4a349da5a03`

## Pinned upstream code

- [`ElSacho/covmetrics`](https://github.com/ElSacho/covmetrics) at `a5205aada6a0f39e3812daf087753217ef66b159`
- [`ElSacho/Conditional_Coverage_Estimation`](https://github.com/ElSacho/Conditional_Coverage_Estimation) at `39a99dcad92205a15d93f2c5fec40c76540abf1c`

The release defects, data provenance, source-versus-caption discrepancy, and
determinism limits are documented in the detailed audit. They are part of the
claim boundary rather than silently repaired assumptions.

## Claim scope

- C1 is a population principle under exact conditional coverage and proper losses.
- C2 is the full seven-method Table-2 statistic over four datasets, ten repeats, and ten test-size levels.
- C3 is the Figure-4 synthetic comparison, with the released generator's scale discrepancy disclosed.
- C4 is the asymmetric ERT decomposition and one-sided construction check.
- C5 is the full Table-4 classification decomposition over four datasets and two strategies.
- C6 is Algorithm 1's held-out k-fold estimator, independently checked on the committed five-seed evidence.

Finite audits support the cited source claims; they do not replace proofs,
universal quantifiers, or the external evaluator.
