FINAL ANSWER

The submitted edit moves all four main readouts toward the true effects, but only by about 6–9% of the gap. Factual fit is preserved: +0.0012 nats per reading across all 400 tuning stays, against a limit of 0.005.

**What I changed.** I edited one tensor, `core.prelude.0.attn.kv.weight`, rows 288–335. These rows are the value projection of attention head 2 in the first prelude layer. On drug bundles this head attends almost entirely to the action token, and in this architecture that token is the only way dose information enters the model's state.
- I amplified this value map only along the directions in which the normalised action token moves when a dose is doubled or halved.
- The insulin directions are scaled by 2.5 and the norepinephrine directions by 8.
- All other directions are unchanged, including the mean action token and the layer-norm bias. So the "a drug is running" signal and everything else the head carries stay as they were.

**Why.**
- My model of the training records, fitted earlier, gives true effects about five times larger than the original model's rollouts. Insulin x2 is about −14 on glucose_finger, against the model's −2.6. Norepinephrine x2 is about +4.3 on map_arterial, against the model's +0.75. These true-effect estimates come from my own fit to the records, not from the tool.
- The model largely ignores dose because this head's output, about 0.7 in size, is swamped by a residual stream of about 800. Amplifying it along the dose directions is the most direct and cheapest fix I found.
- The other routes I tried did worse:
  - readouts built on the prediction heads;
  - an extra channel injected into the core;
  - scaling the dose embeddings;
  - scaling the whole head uniformly.
- The strength of the edit is set by the factual-fit limit:
  - Stronger versions did much better on effects. At 12/12, insulin x2 reached −9.2 but cost +0.017 nats.
  - Cost grows faster than the gain does.
  - Cost on general stays was about four times the cost on my drug-selected test stays.
- So I chose the final strength by checking the cost on all 400 tuning stays.

**Factual fit.** The value NLL change is +0.0012 nats per reading over all 400 tuning stays. The four 100-stay subsets came in at +0.0029, +0.0018, −0.0002 and +0.0002, so even the worst is well under 0.005.

**Expected readouts.** Measured with the tool's dose_edit rollout on 75 tuning stays:

| Readout | Original | Edited | True effect (from the records) |
|---|---|---|---|
| Insulin x2 → glucose_finger | −2.65 | −3.70 | about −14 |
| Insulin x1/2 → glucose_finger | +1.27 | +1.71 | about +7 |
| Norepinephrine x2 → map_arterial | +0.75 | +0.95 | about +4.3 |
| Norepinephrine x1/2 → map_arterial | −0.44 | −0.59 | about −2.1 |

- The other scored interventions should move the same way by similar small fractions: stopping the drug, x1.5, x3, starts at hours 6 and 16, and the glucose_lab and map_cuff readouts.
- The change in total log loss over all outputs should be small, because the edit only slightly shifts the shared state.

The remaining gap is large. With this mechanism, reaching the true effects would cost far more factual fit than the 0.005 limit allows.