# ADR-004: What this repository deliberately cannot measure

## Status

Accepted.

## Context

A tool that reports a stability verdict invites a reader to treat that verdict as the last
word. It is not, and the honest boundaries are worth stating as a decision rather than
leaving to be discovered.

## Decision

Record, and link from the README, the four things the repository does not establish.

1. **Within-dataset resampling is not across-dataset sampling.** The bootstrap reuses the
   rows of the one dataset in hand, so it measures reproducibility under resampling, not
   under drawing a fresh dataset. Two features that are tied in the population can look
   stably ordered within a single dataset. Experiment 4 shows this directly: across tied
   datasets the single fit names a winner about half the time, yet within one dataset the
   bootstrap still orders the tied pair confidently in 18% of cases.
2. **It assumes a linear model and independent features for the attribution.** The exact
   closed form (ADR-001) is the Shapley value under feature independence. Where features
   are dependent, the interventional and conditional attributions differ, and neither is
   "the" SHAP value. The tool measures the stability of the independence-case linear
   attribution, not its correctness.
3. **It does not transfer to tree or kernel SHAP unchanged.** Tree SHAP has its own
   variance structure across refits; the mechanism here (coefficient variance under
   collinearity) is specific to the linear case. The method would carry over, the numbers
   would not.
4. **The fixture is synthetic.** The groups are planted so the ground truth is known. This
   makes the calibration claims exact, and it means the specific figures are facts about
   the generator, not about any real dataset. The CLI accepts a CSV so the same analysis
   runs on real data, where the ground truth is unknown.

## Consequences

The verdict should be read as "is this ranking reproducible under resampling of this
dataset, for a linear model", with the three further limits above. That is a narrower and
more defensible claim than "these feature importances are real", and stating it is what
makes the claims the tool does make believable.
