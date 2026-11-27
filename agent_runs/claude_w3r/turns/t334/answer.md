The factual pass is decisive and refutes my local estimate: the saved edit costs **+0.134 nats** overall (gf +0.147, map_arterial +0.332) — 27x over the 0.005 budget. My offline `QG`/`QM` panels evidently did not faithfully reproduce real `with_code` rows, so the NLL improvement I "verified" locally was an artefact.

Combined with the measured intervention-SNR, this settles it: the cost of this channel grows quadratically with gain while the dose response grows linearly. Scaling to a budget-safe level (≤1.2/7 of current) would leave an insulin effect of ~0.06 mg/dL — indistinguishable from the unedited model, while risking constraint failure on a single extrapolated estimate.

With one Python run left and no ability to revert afterwards, the correct move is to drop all edits and submit the original weights, whose factual fit is guaranteed to pass.

```python
for nm in list(edited()): drop_edit(nm)
print('edited now:', edited())
open('/work/notes.md','a').write(
 "\n## Session 16 final\n"
 "- fwd_7 factual(both,75 tuning): orig 1.63425 vs edited 1.76784 => +0.134 nats (gf +0.147, map_art +0.332, map_cuff +0.020). 27x over the 0.005 budget.\n"
 "- => QG/QM offline panels are NOT faithful to real with_code rows (they predicted a gf NLL IMPROVEMENT; truth was a large worsening). All gain calibration based on them is invalid.\n"
 "- Measured channel economics: cost ~ scale^2, dose-effect ~ scale. At scale 7: effect contribution ins -0.32 mg/dL, cost +0.134 nats.\n"
 "  Budget-safe scale <=1.2 => ins effect ~-0.06 mg/dL, i.e. nil. Root cause: intervention-SNR of any linear read of q is ~1/23\n"
 "  (dose doubling moves the read 0.011 vs cross-sectional sd 0.25), because the action-driven component of h is attenuated by out_norm.\n"
 "- A linear reader of the state cannot form (rate - E[rate|state]) (the 'deviation form' that is NLL-free): both terms are the same\n"
 "  linear functional of q. A correct edit needs the action-driven component amplified in the core BEFORE out_norm (Plan A), which\n"
 "  needs the prelude->b_in channel; my built version (b_in delta norm 2.4e4) wrecked the state (fwd_4: ins effect exactly 0.0).\n"
 "- Local simulation was lost: run 293 clobbered the kernel globals B/rows/EU that stay_data() closes over; restoring B from train.csv\n"
 "  did not fix it, so step_at/stay_data are unusable and Delta-q could not be measured.\n"
 "- DECISION: dropped all edits; submitting the ORIGINAL weights. Guarantees the factual constraint (0.000 nats change).\n"
 "  Readouts will be the original model's: ins x2 -0.52, ins x1/2 +0.05, ne x2 -1.74, ne x1/2 -0.64\n"
 "  (targets -13.71 / +6.86 / +5.17 / -2.58). Honest but poor on the causal readouts.\n"
 "- For a future attempt, the order should be: (1) FIRST validate any offline q-panel against a factual forward pass before using it\n"
 "  to calibrate gains; (2) build the amplification inside the core (pre-out_norm) so the action-driven signal is not attenuated;\n"
 "  (3) never overwrite kernel globals - use prefixed names only.\n")
```