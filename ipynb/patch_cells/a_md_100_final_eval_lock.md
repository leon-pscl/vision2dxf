### The single-use lock on Test-A and Test-B

Test-A and Test-B may each be read once, and this is the only place that reads them. That makes the
lock mechanical rather than a matter of discipline: `evaluate_test_lock` compares the SHA-256 of
each variant's `best.pt` against the hashes recorded in the previous `final_eval.json`, and takes one
of three actions.

| situation | action | consequence |
|---|---|---|
| no previous `final_eval.json` | `run` | the single-use evaluation happens now |
| previous record, **same** weight hashes | `reuse` | the cached tables are reloaded; the test sets are not read again |
| previous record, **different** weights | `blocked` | the evaluation is refused |
| as above, `allow_reeval_after_change: true` | `run` | re-evaluated, and the model card records that the override was used |

`reuse` exists because a re-run of the notebook with nothing changed would otherwise spend the
single-use budget again — the brief reserves Test-A and Test-B for *one* evaluation, and repeating
it from identical weights is still a second look.

`blocked` exists because re-evaluating from changed weights is selection on the test set: once you
have read Test-A, changing the model and reading it again is how you stop having a held-out set. So
the lock refuses, prints what changed, and points at the override.

On `blocked` the notebook **does not stop**. Sections 9–11 still run and still write a bundle,
whose model card records that the test metrics are absent and why. Refusing to evaluate should not
mean refusing to produce anything else.

If either production variant is unavailable — a training phase was blocked, so there is no model to
evaluate — the test sets are not read at all, and the reason is printed.