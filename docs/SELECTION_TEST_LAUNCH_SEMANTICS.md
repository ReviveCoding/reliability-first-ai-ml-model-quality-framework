# Selection, Untouched Test, and Launch Readiness

State flow:

candidate models
-> selection winner
-> promotion candidate
-> untouched-test launch gate
-> approved champion only on PASS

Artifacts:

- `model_selection_decision.json`: selection-window decision only.
- `candidate_test_evaluation.json`: untouched-test metrics for the selected candidate.
- `portfolio_test_diagnostics.json`: diagnostic only; never used to reselect.
- `launch_gate_result.json`: launch decision for the promotion candidate.

`REVIEW` and `BLOCK` must leave `approved_champion` as null.
