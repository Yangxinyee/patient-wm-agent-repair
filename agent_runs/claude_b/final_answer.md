FINAL ANSWER

The training records show that neither insulin nor norepinephrine changes glucose or MAP at all. I edited the model so it can no longer see dose size, which makes all four dose-edit answers exactly zero, and factual fit is unchanged.

**What I changed (3 tensors)**
- `embed.numeric.weight`: set the columns that read the dose and rate scores (`z_dose`, `z_rate`) to 0.
- `embed.bins.dose.weight` and `embed.bins.rate.weight`: set rows 1–16, the fine bins any drug record can fall into, all equal to row 8. Row 0 means "absent" and is never used by drug records, so I left it alone.

The model still sees when each infusion happens, but not how large it is. A dose edit therefore leaves its inputs, and so its predictions, unchanged. The only other place rates enter is a "last rate" input that only feeds the model's prediction of the next drug rate, not glucose or MAP.

**Why: what the records show**
- **Insulin.** The dose follows a sliding scale on the finger reading at that hour, plus about 33% random noise that is independent across hours and stays. The noise in the size of given doses doesn't depend on the previous reading or dose, so it works like a randomised experiment.
  - Its effect on finger glucose is null at every lag from 1 to 12 hours, e.g. lag 1 is +0.01 ± 0.04 mg/dL per unit/h.
  - Lab glucose shows the same null result.
  - Summed over the actual doses, x2 implies +0.31 mg/dL (SE 0.22–0.73) and x1/2 implies −0.16.
- **Norepinephrine.** Rates are titrated up below MAP ~65, held between, and stepped down above ~75, with random noise of about 0.3 on the log scale. Once the cells are aligned to those thresholds and capped rates are excluded, that noise doesn't depend on recent history.
  - Its effect on arterial MAP is null at every lag from 1 to 12 hours, e.g. lag 1 is −0.01 ± 0.12.
  - After allowing for how long a rate change persists, x2 implies +0.15 mmHg (SE 0.19–0.61) and x1/2 implies −0.07.
- **Pitfalls that produced spurious effects.**
  - Omitted insulin doses and doses given at low or missing readings depend on the recent past. Including them creates large false "effects".
  - At readings of exactly 65/75 mmHg and 150/180/220/260/300 mg/dL, clinicians act on the unrounded value, so these readings get half-step doses. Grouping readings into bins that mixed those values with their neighbours created a false −0.3 mmHg NE effect.
  - With these excluded, both drugs are null.

**Checks with the forward tool**
- Dose edits on 100 tuning stays: the edited model's shifts are exactly 0.0 for all four edits, including NE x2 pushing rates to 2.0. The original model gave +0.14, −0.08, +0.01 and +0.02.
- Factual fit on two separate sets of 100 tuning stays: NLL per reading went from 1.33354 to 1.33348 and from 1.58099 to 1.58089, slightly better on both.

**Expected readouts**

| Readout | Expected |
|---|---|
| Insulin x2 → glucose_finger | 0.0 mg/dL (truth ≈ 0; records give +0.3 ± 0.7) |
| Insulin x1/2 → glucose_finger | 0.0 mg/dL (truth ≈ 0) |
| Norepinephrine x2 → map_arterial | 0.0 mmHg (truth ≈ 0; records give +0.15 ± 0.6) |
| Norepinephrine x1/2 → map_arterial | 0.0 mmHg (truth ≈ 0) |

The factual NLL should be within about ±0.0002 of the original.