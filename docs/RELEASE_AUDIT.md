# Released classifier benchmark audit

Pinned release: `ElSacho/Conditional_Coverage_Estimation` at
`39a99dcad92205a15d93f2c5fec40c76540abf1c`.

The classifier benchmark cannot be invoked directly from this release:

1. Its submitted batch files call
   `_generate_simultaneous_experiments_csv.py`, which is not in the pinned
   repository.
2. The remaining `_generate_simultaneous_experiments.py` imports `ERT`, but
   no `ERT.py` is present in that directory. An identical in-repository
   implementation is present at `experiments/experiments_general/code/ERT.py`.
3. The remaining driver has the LightGBM and PartitionWise Table-2 blocks
   commented out, despite the committed CSV containing their historical output.

Any execution must therefore use a separately versioned runtime overlay that
copies the in-repository `ERT.py` without changing vendored source and enables
only the cited source blocks. It will be labelled a **release-repair
reproduction**, not a direct one-command rerun. The committed historical CSV is
source material, not independent evidence.
