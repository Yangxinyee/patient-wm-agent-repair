FINAL ANSWER

Kept the four edited value-bin head tensors, calibrated from training-record dose-response estimates while constraining factual loss. All other weights remain original; no gradients were used.

Expected readouts, based on 400 tuning stays:

| Intervention | Predicted mean shift |
|---|---:|
| Insulin ×2 → glucose_finger | −10.00 mg/dL |
| Insulin ×½ → glucose_finger | +5.22 mg/dL |
| Norepinephrine ×2 → map_arterial | +1.38 mmHg |
| Norepinephrine ×½ → map_arterial | −1.32 mmHg |

Factual value NLL worsened by **0.003342 nats/reading**, below the 0.005 limit on tuning.

The insulin response approaches the records-based estimate. The norepinephrine repair remains partial. These predictions are not known intervention truth, and the additional scales, start times, and cross-sensor readouts remain unvalidated.