# ADR-001: Exact closed-form SHAP for a linear model, not the shap library

## Status

Accepted.

## Context

The repository measures whether a SHAP feature ranking is reproducible. To do that it has
to compute SHAP importances many thousands of times, once per bootstrap refit. Two paths
were available: call an existing explainer such as the `shap` library (kernel SHAP, or its
LinearExplainer), or compute the attribution directly.

## Decision

Compute the exact linear attribution in a few lines of numpy. For a linear model with the
standard feature-independence assumption, the Shapley value of feature i on row x is
beta_i times the centred feature value, which on standardised, centred features is simply
beta_i times z_i. Global importance is the mean absolute attribution over a fixed
evaluation set, which factorises exactly into the absolute coefficient times a fixed
per-feature constant.

## Alternatives considered

- **Kernel SHAP from the `shap` library.** It is model agnostic and would let the tool
  wrap any estimator, but it is an approximation with its own sampling noise. That noise
  would be indistinguishable from the coefficient instability the tool is trying to
  isolate, which defeats the measurement. It is also a heavy dependency and, for the
  linear case, computes by sampling what is available in closed form.
- **The library's LinearExplainer.** Correct for this case, but it still pulls in a large
  dependency and its internals would have to be pinned to keep the numbers reproducible.
  The closed form is six lines and fully owned.

## Consequences

The runtime dependency is numpy alone, every importance is exact to machine precision, and
all run-to-run movement in importance is attributable to the coefficient rather than to an
explainer's own randomness. The cost is scope: this is a linear-model tool. Extending it
to tree models would require exact tree SHAP, which is a different computation and is out
of scope by ADR-004. The independence assumption behind the closed form is also a
boundary, recorded in ADR-004.
