# Experimental Evidence

## End-to-end validation

The framework completed CPU smoke, GPU smoke, full GPU training and
inference, semantic finalization, launch-gate evaluation, and Monte Carlo
scenario validation.

## Transformer stability audit

The unweighted Transformer was evaluated with seeds
`7, 17, 27` on an identical frozen
19,996-row dataset. The final stability verdict
was `STABLE`.

Mean baseline test Macro-F1:
`0.605731`.

Mean baseline test worst-slice F1:
`0.311275`.

## Class-weighted challenger

The sqrt-balanced challenger used train-only class weights and the same
seeds, frozen data, checkpoint-selection split, framework-selection
split, and untouched test protocol.

It improved mean selection Macro-F1 by `+0.033361` and mean
test Macro-F1 by `+0.015000`, but changed mean selection
worst-slice F1 by `-0.004329` and mean test
worst-slice F1 by `-0.037691`.

The challenger was therefore rejected under the predeclared criteria.

## Claim boundary

All results are local, offline, portfolio evidence based on public or
synthetic data. They do not claim production deployment, private
customer-data access, or Apple-internal validation.
