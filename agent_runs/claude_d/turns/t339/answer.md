FINAL ANSWER

My edit makes the model's insulin dose answers match the effect I estimated from the training records. Its NE answers have the right sign but are too small, and the factual fit stays well inside the budget.

| Readout | Original model | Edited model (sel1 rollout) | My estimate of the true effect |
|---|---|---|---|
| Main: insulin x2 → glucose_finger, hours 13–24 | −1.7 | −13.8 mg/dL | about −13 to −14 |
| insulin x1/2 → glucose_finger | +0.9 | +7.0 mg/dL | about +6.5 to +7 |
| norepinephrine x2 → map_arterial | −0.1 | +0.7 mmHg | about +4 to +5 |
| norepinephrine x1/2 → map_arterial | −0.5 | −1.1 mmHg | about −2 to −2.4 |

On a second set of 100 tuning stays, an earlier version with the same insulin channel gave −13.4 and +6.1, against estimates of −13.2 and +6.6. The "true effect" column is my estimate from the records, not the actual simulator.

**Factual fit.** I measured the change in value NLL per reading on all 400 tuning stays, in four sets of 100. The results were +0.0039, −0.0006, +0.0016 and −0.0011, about +0.0010 pooled. The budget is 0.005.

**What the records show.** In the original model, glucose readings that follow unexpectedly high insulin come out lower than it predicts, by about 1.3 mg/dL per u/h of surprise insulin one and two hours later. The model learned how doses go with the patient's condition, not what insulin does. Unexpected norepinephrine is followed by higher MAP, about +3 mmHg per unit one hour later.

**What I changed.** Only weights, no gradient training:

1. **Insulin channel in the value head** (`attr_heads.value_bin.0` hidden units 31 and 143, and their columns in `value_bin.2.weight`).
   - The pair switches on only for a glucose reading when insulin ran in the previous hour.
   - It then reads my linear estimate of how much insulin differed from what the history predicted, and moves the glucose prediction accordingly.
   - I set its strength by matching the dose_edit rollout to my estimated effect.
   - It is built so that the rollout's lower-precision (bf16) arithmetic does not distort it.
2. **Norepinephrine channel** (hidden units 124 and 142). It has the same design, for map_arterial readings when norepinephrine ran. Its strength is set to 0.4. A small change to `value_bin.2.bias` compensates for taking these units away from their original role.
3. **Encoder fix for doubled norepinephrine** (`encode_action.mlp1`, four hidden units).
   - Recorded norepinephrine rates stop at 1.0, so doubling pushes many doses into a top bin the model never saw for this drug, and it ignored them.
   - The fix makes such doses look like rate 1.0 plus a straight-line extension of the in-range dose effect.
   - These doses never occur in the records, so this costs no factual fit.

**Why norepinephrine is under-predicted.** My first version had both channels at full strength. On tuning stays it cost +0.010 in factual NLL, double the budget. The norepinephrine channel's corrections on factual readings were almost pure noise, so I scaled it to 0.4. Going higher (0.6–0.7) would put the stays heaviest in drug use over the limit, for less than 1 mmHg of gain.

The x2 estimate is also the least certain of my targets: most of it comes from rates above anything recorded, where I assumed the effect keeps rising in a straight line.