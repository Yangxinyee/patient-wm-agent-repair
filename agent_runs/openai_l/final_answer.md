FINAL ANSWER

Kept the conservative seven-tensor edit: three dose-processing/attention tensors and four value-head tensors. The corrections use training-record evidence and closed-form statistical fits, without gradients or medical assumptions.

Expected readouts, based on the latest paired tuning rollout:

| Intervention | Predicted mean change |
|---|---:|
| Insulin ×2 → glucose_finger | −10.20 |
| Insulin ×½ → glucose_finger | +4.92 |
| Norepinephrine ×2 → map_arterial | +2.42 |
| Norepinephrine ×½ → map_arterial | −1.15 |

These are model predictions, not observed counterfactual truth. Insulin responses approach the provisional record-derived targets; norepinephrine remains underestimated.

Across all 400 tuning stays, factual value NLL increased **0.00419 nats/reading**, below the allowed 0.005 in this audit. Sampling uncertainty means this does not guarantee the unseen-stay constraint. All-output log loss and unsupported intervention variants remain unverified.