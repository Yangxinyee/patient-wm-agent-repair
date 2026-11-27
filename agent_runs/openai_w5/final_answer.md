FINAL ANSWER

Retained the weight-only dose-response repair: record-derived drug coefficients, private drug-memory channels, factual centering, and **1,100-second insulin persistence**. No gradient training was used.

The latest 100-stay paired rollout predicts these shifts in native reading units:

| Intervention | Readout | Predicted shift |
|---|---|---:|
| Insulin ×2 | glucose_finger | −4.221 |
| Insulin ×½ | glucose_finger | +2.241 |
| Norepinephrine ×2 | map_arterial | +3.040 |
| Norepinephrine ×½ | map_arterial | −0.856 |

These responses are substantially stronger than the original model’s near-zero responses. Training-record regressions motivated the repair; they remain assumption-dependent estimates, not observed intervention truth.

**Validation limitation:** the supplied restart summary does not contain the final candidate’s pooled factual-NLL result. I therefore cannot certify the +0.005-nat guardrail, all-output log-loss preservation, or generalization to the additional interventions.