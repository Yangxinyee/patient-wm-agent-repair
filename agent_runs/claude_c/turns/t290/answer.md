FINAL ANSWER

The edit gets the dose answers moving the right way, but they remain far too weak. The model now shows about a fifth of the true insulin effect and very little of the norepinephrine effect.

**What I changed.** Two rank-1 operations on `embed.numeric.weight`, side "in", alpha 4, along the z_dose and z_rate input columns. This multiplies those two columns by 5, so each drug record's dose and rate count five times as strongly in its embedding.

**Why.** From the training records:
- Insulin's true effect can be measured cleanly. Within a glucose tier the recorded rate varies only by independent random noise. That noise shows about −2 mg/dL of finger glucose per U/h one hour later, about −11 in total over 12 hours, and the effect is linear in dose.
- Norepinephrine raises arterial MAP by about +3.4 mmHg per mcg/kg/min one hour later, about +9 in total, also linear.
- On those same stays the true readout changes come out at about −13.6 (insulin x2), +6.8 (insulin x1/2), +4.5 (norepinephrine x2) and −2.2 (norepinephrine x1/2).
- The original model largely ignores dose. Its factual predictions show almost no slope on the dose noise. In rollouts it gives about 1/7 of the true insulin effect, and it gives a negative MAP change for both norepinephrine edits.

**Why only this much.** Dose information reaches predictions only through the action token. That token is layer-normalised twice before the prelude attention, and its total pull on predictions is small: removing it shifts glucose predictions by about 1.1 mg/dL on average. Raising the dose input rotates the token and hits a ceiling: x5 worked best, x20 was weaker.
- **Ruled out (not the dose path):** the z-token and z-posterior routes, edits to the state matrices and output/readout directions, and the bin rows.
- **Too costly in factual fit:** stronger attention output overall, or a boosted attention head. These also amplify non-drug information, costing +0.004 to +0.017 nats.
- **No new lever on recheck:** the only drug-specific tensors are constant embedding rows, which can rotate the token but not strengthen it.

**Expected readouts** (model, from 100–200 tuning stays; estimated truth in brackets):

| Readout | Expected model value | Estimated truth |
|---|---|---|
| Insulin x2, glucose_finger | about −2.3 to −2.5 (original −1.9) | −13.6 |
| Insulin x1/2, glucose_finger | about +1.7 to +1.8 (original +0.8) | +6.8 |
| Norepinephrine x2, map_arterial | about +0.15 (original −0.3 to −0.5, wrong sign) | +4.5 |
| Norepinephrine x1/2, map_arterial | about −0.1, range −0.24 to +0.06 (original −0.6) | −2.2 |

Three readouts move closer to the truth. Norepinephrine x1/2 moves slightly away, because the original's −0.6 came from an artefact that also made its x2 answer negative.

**Factual fit:** on 300 tuning stays the value NLL rises by +0.0021 nats per reading (standard error 0.0008), within the 0.005 limit. Glucose_lab bears most of the cost, while map_arterial improves slightly.