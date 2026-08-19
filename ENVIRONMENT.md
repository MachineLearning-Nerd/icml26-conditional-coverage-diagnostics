# Reproduction environment

## Pinned setup

- Python: `3.11.*`
- Package manager: `uv` with the committed `uv.lock`
- Platform target: CPU-first; the Claim-6 checker uses only the Python standard library
- Source metric package: `covmetrics` pinned to `a5205aada6a0f39e3812daf087753217ef66b159`
- Upstream experiment release: `ElSacho/Conditional_Coverage_Estimation@39a99dcad92205a15d93f2c5fec40c76540abf1c`
- The lockfile includes the explicit CPU PyTorch index for Linux and the additional `numba` dependency required by the released calibration path.

## Reproduce the independent current audit

```bash
uv sync --frozen
python3 repro/src/audit_claim6.py
```

The expected result is `CLAIM 6 VERIFIED`, 270 recomputed summary values, and
three rejected negative controls. The committed output is the evidence record;
the command does not create a live evaluator score.

## Full protocol boundary

The source-faithful Table-2 and Table-4 routes are available in the repository,
but the full literal protocols are not claimed as locally rerun within the
authorized two-hour CPU envelope. TabPFN/TabICL are optional foundation-model
comparators and are never used as evidence for a CPU result. Data and generated
outputs are intentionally excluded from the repository where the source release
does not provide them; their provenance and hashes are recorded in the source
audit.
