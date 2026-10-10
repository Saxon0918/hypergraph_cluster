# E1/E3 performance change record

## E1

- Exact spectra and third-order moments now reuse one validated dense Gram
  operator per real or null object.  Mathematical definitions and all reported
  moment decompositions are unchanged.
- The N2 proposal distribution is unchanged: two distinct incidence slots are
  sampled uniformly and rejected moves still count.  Candidate pairs are drawn
  in batches, edge fingerprints use exact `frozenset` membership, and duplicate
  checks no longer copy the complete fingerprint set on every proposal.
- Independent N1 batches and the four independent N2 chains run in processes.
  Each process is constrained to the configured number of BLAS threads.
- Partial and complete sample checkpoints are atomic.  Final CSV/Parquet files
  are rebuilt by streaming checkpoints instead of retaining all curve-row
  dictionaries in memory.
- The predeclared scale protocol is applied: classification mechanism samples
  share the dataset median mean edge size; selected Congress windows share the
  median mean edge size; other objects retain their own size-only scale.

## E3

- Every graph still contributes to the forward and inverse approximation error
  tables.  Runtime benchmarking uses a deterministic, dimension-spanning subset
  per dataset, recorded in `benchmark_manifest.csv`.
- Within each timed repetition, operator construction and spectrum/moment
  computation occur once.  Their result is reused for `J=1,12,80`, as required
  by the implementation specification.
- Operator, numerical computation, query, and total times are stored separately.
  Very small query calls are repeated and divided by the loop count.
- Direct Exact-G is marked `not_run` above the configured edge-side dimension
  limit; Exact-small-side, Moment-3, and approximation-error analysis continue.
- Samples run in bounded process-level parallelism with one BLAS thread per
  worker, per-sample atomic checkpoints, progress, and resume support.
