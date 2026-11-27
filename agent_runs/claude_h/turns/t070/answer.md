FINAL ANSWER

**Conclusion.** In this ICU, as far as the 30,000 training stays can show, neither drug changes the readings it targets. I edited the model so that its glucose and MAP predictions no longer depend on how much insulin or norepinephrine was given. All four dose readouts are therefore exactly 0, and the factual fit is unchanged.

**What I changed (3 tensors):**
- **`embed.bins.dose.weight` and `embed.bins.rate.weight`:** rows 1–16, the bins for a recorded dose or rate, are all set to the original row 8 (the median bin). Row 0, which means "no value", is untouched.
- **`embed.numeric.weight`:** the `z_dose` and `z_rate` columns are set to zero. The other columns are untouched.
- **Effect:** the model still sees whether and when each drug was given, but not the amount.
- **No other route for dose:** the per-code "last value" register also carries rates, but only for drug codes. Glucose and MAP predictions never use a rate, so the readings' predictions cannot depend on dose in any stay.

**Why: what the records show.**
- **Insulin policy:** insulin follows a sliding scale on the recorded finger glucose (lab glucose when there is no finger reading). The steps are at 150/180/220/260/300 mg/dL. Each dose is multiplied by a random factor between 0.5 and 2, independent from hour to hour and of the patient's history. That random factor is a natural experiment.
  - Within narrow glucose bands, the dose has no measurable effect on glucose 1 to 12 hours later (standard error 0.04–0.11 mg/dL per u/h).
  - Comparing readings just above and just below each sliding-scale step also shows no effect.
  - The implied insulin x2 readout is +0.9 ± 0.5 mg/dL on finger glucose and +0.3 ± 0.6 on lab glucose. As a placebo check, insulin's estimated effect on MAP is 0.
- **Norepinephrine policy:** norepinephrine is titrated on the recorded MAP, with a random hour-to-hour adjustment (sd 0.14).
  - The policy uses the recorded values themselves: arterial when present, and with the same response to the noisier cuff readings. So the clinician has no hidden information, and the random part identifies the effect cleanly.
  - The implied norepinephrine x2 readout is about −0.3 ± 0.14 mmHg. It shifts between analysis choices and is about 0 after hour 12. That is far below any meaningful effect, so I treat it as absent.
- **Original model:** it already gave readouts near 0 (insulin x2 +0.05 to +0.07, x1/2 −0.03 to −0.15; norepinephrine within ±0.03). Those values were rollout noise. The edited model gives exactly 0.

**Checks with the forward tool (4 of 50 passes used):**

| Check | Original | Edited |
|---|---|---|
| Dose readouts, 100 tuning stays (39 insulin, 5 norepinephrine stays) | about 0, noisy | 0.0 at every stay-hour |
| Factual NLL, set 1 | 1.34364 | 1.34369 (+0.00005) |
| Factual NLL, set 2 | 1.58471 | 1.58464 (−0.00007) |

Both factual differences are far inside the 0.005-nat allowance.

**Expected readouts on the evaluation stays:**

| Readout | Edited model | True value I expect |
|---|---|---|
| Insulin x2 → glucose_finger | 0.0 mg/dL | ≈ 0 |
| Insulin x1/2 → glucose_finger | 0.0 mg/dL | ≈ 0 |
| Norepinephrine x2 → map_arterial | 0.0 mmHg | ≈ 0 |
| Norepinephrine x1/2 → map_arterial | 0.0 mmHg | ≈ 0 |

If the faint signals were real, the true values would still be tiny: about +1 mg/dL for insulin x2 and −0.3 mmHg for norepinephrine x2. The halving edits would be roughly half those sizes, with the opposite sign. The factual NLL should stay within about ±0.0001 of the original.