# ADR-002: Manufacture the missing variance by bootstrapping the training rows

## Status

Accepted.

## Context

A single fit yields one importance vector and no error bar, so the ranking it produces
cannot be told from one that reordered by luck. To attach a distribution to the ranking,
the tool has to refit under some source of variation. The choice of source decides what
question the stability verdict actually answers.

## Decision

Resample the training rows with replacement (the nonparametric bootstrap) and refit on
each resample, holding the evaluation set and the standardiser fixed. The spread of the
resulting importances is the sampling variability of the ranking: how much the order would
move if the same data-generating process had handed over a different sample of the same
size.

## Alternatives considered

- **Model-seed-only variation** (refit with a different random seed but the same rows).
  For the closed-form ridge there is no seed, so this would measure nothing. Even for a
  stochastic learner it would capture only optimiser jitter, not the sampling variability
  that actually moves a ranking.
- **Generating fresh datasets.** This would measure across-dataset variability, which is a
  stronger and different notion. It is not available to someone who has one dataset in
  hand, which is the whole situation the tool is built for. The gap between the two is real
  and is measured and documented in experiment 4 and ADR-004, rather than papered over.
- **Refitting the standardiser per resample.** Rejected because it injects evaluation-set
  noise on top of coefficient noise. Fixing the standardiser and the evaluation matrix is
  what lets the verdict attribute instability to the coefficient alone.

## Consequences

The verdict answers a precise question: is this ranking reproducible under resampling of
this dataset? That is the right question for a practitioner with one dataset, and the
honest limit is that a pair of features that are tied in the population can still look
stably ordered within a single dataset, because resampling reuses the same rows. The war
story in the README records a related trap: an early version centred only y, so each
resample's non-zero column means leaked into the coefficients and added a confound to this
very variance.
