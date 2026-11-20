# ADR-003: Judge a ranking by pairwise dominance, a reliability threshold, and a floor

## Status

Accepted.

## Context

Once there is a distribution over importances, it has to become an answer a reader can
act on. A ranking is used one comparison at a time ("A matters more than B"), so the
natural unit to judge is the adjacent comparison, and the natural statistic is the
dominance probability: the share of resamples in which the higher-ranked feature really is
the more important one.

## Decision

For each adjacent pair inside the judged depth, estimate the dominance probability and put
a two-sided Wilson interval on it. Grade the boundary:

- **resolved** if the interval sits above a reliability threshold (the higher feature wins
  at least 90% of the time);
- **weak** if the interval clears 0.5 but not the threshold;
- **tie** if the estimate is within a small band of 0.5;
- **reversed** if the interval sits below 0.5;
- **underpowered** if the interval still straddles 0.5 but the estimate is off it.

The ranking is STABLE only if every boundary is resolved; a tie, a reversal or a weak
boundary makes it UNSTABLE; an underpowered boundary alone makes it INCONCLUSIVE. Report a
resolution floor of one over the square root of the budget, the smallest dominance gap the
resample count can resolve, next to every verdict.

## Alternatives considered

- **Significance against 0.5 alone.** This was the first design, and it was wrong. At 500
  resamples a dominance of 0.54 is significantly above a coin flip yet means the order
  flips almost half the time. The reliability threshold was added precisely because
  significance is not trust: a 54/46 order should not be called stable.
- **A single stability score** (for example the share of resamples reproducing the exact
  top-k order). It is a vivid number, and it is reported, but it is too strict to be the
  verdict: near-tied features break exact order even when every decision-relevant
  comparison is reliable.

## Consequences

The verdict is conservative in the right direction: it refuses to bless an order that holds
only weakly. The resolution floor makes the budget's reach explicit, so an inconclusive
boundary comes with the number of resamples that would settle it, rather than looking like
a failure of the tool. On the default data the floor is 0.045 at 500 resamples, and the
one underpowered boundary would need about 831 resamples to resolve.
