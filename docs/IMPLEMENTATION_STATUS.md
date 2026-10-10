# Implementation status

## Verified in the requested interpreter

The gold-standard graph returns `W=(3,8,34,176)`, decomposition
`(A2,O2,A3,B3,T3)=(22,12,62,102,12)`, the specified cumulants, eigenvalues,
and beta=0.1 thermodynamic values to float64 tolerance. G/Q equivalence,
monotonicity, exact inversion, loaders, N1/N2 invariants, simultaneous school
cliques, structural grouping, local feature extraction, nested CV, E1, E3, E4,
Parquet output and PDF/PNG reporting were exercised with synthetic fixtures.
The optimized E2 pipeline was also exercised end-to-end on both configured HIC
datasets with the quick protocol: fold-level feature reuse, lazy NetLSD caches,
LinearSVC, progress output, runtime accounting, and method checkpoints all
completed.  This remains a pipeline/performance validation, not the formal E2
result; the full five-seed nested protocol must still be run on the server.
The optimized E1/E3 paths were exercised end-to-end with the smoke fixtures,
including process-level parallelism, one-thread BLAS limits, joint Gram reuse,
staged E3 timing, streamed result assembly, and interrupted-run checkpoints.
On the largest selected Congress N2 object, the optimized switch kernel was
also checked for invariants after a real-data proposal batch.  These checks are
implementation validation, not formal E1/E3 scientific results.

## Requires official data or external code before scientific reporting

- HIC real-data byte identity and post-cleaning counts.
- Enron/Congress full archives and verified calendar origins.
- Primary-school ID mapping and verified local clock/calendar alignment.
- Formal N2 convergence/ESS assessment at real-data scale.
- B8/B9 author HIC checkouts and their pinned adapter command.
- Optional MoCHy, RethinkHGK, and GNN extensions.
- Full E1-E4 runs, their compute budget, and all scientific conclusions.

Missing items are emitted as `not_run`, a failure status, or a manifest gap. No
placeholder values are considered results.
