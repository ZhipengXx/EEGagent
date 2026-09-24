# V1.4 diagnostic agent

Claim scope remains `numeric_consistency_only`.

The planner prompt is `planner_v2`. A tool success does not close a question unless the signal mode and metric keys match the contract. Raw temporal diagnostics do not answer the contrast event-window question.

StopGate may return `pass_configured` after required descriptive questions are done and no heuristic ticket is open. An open required follow-up with no remaining action becomes `abstain` and `coverage=partial`. Failed replan with open questions is `planning_blocked`.

Heuristic routing (`exploratory_v1`, threshold 5.0 on `max_step_over_median`) only requests a closer look. `defect_confirmed` stays false after a targeted description.
