The active-dose decoder has useful held-out signal, but makes large errors in inactive hours. The separate infusion classifier is much more reliable, so I’ll test a gated correction: it should act only on glucose readings when insulin is detected. I’ll then validate the weights directly. Better insulin rollouts with NLL deterioration no greater than 0.005 would support retaining it.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
from pathlib import Path

def trial_gelu_np(x):
    return F.gelu(torch.as_tensor(x, dtype=torch.float64)).numpy()

trial_dfit = trial_active_decoder
trial_cfit = max(trial_classifiers, key=lambda pair: pair[0][1])[1]
trial_s_va = trial_dfit.predict(trial_X[trial_va])
trial_m_va = trial_cfit.decision_function(trial_X[trial_va])
trial_eps = 0.02
trial_gate_candidates = []
for steepness in [5.0, 10.0]:
    for threshold in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        gate = steepness * (trial_m_va - threshold)
        pred = (trial_gelu_np(gate + trial_eps * trial_s_va)
                - trial_gelu_np(gate - trial_eps * trial_s_va)) / (2 * trial_eps)
        result = (
            r2_score(trial_y[trial_va], pred),
            steepness, threshold,
            np.sqrt(np.mean(pred[~trial_active[trial_va]] ** 2)),
            np.std(pred[trial_active[trial_va]])
        )
        trial_gate_candidates.append((result, pred))
trial_gate_result, trial_gated_va = max(trial_gate_candidates, key=lambda pair: pair[0][0])
print("Gated held-out R2, steepness, threshold, inactive RMS, active prediction SD:")
print(trial_gate_result)
print("Ungated held-out all-hour R2:", r2_score(trial_y[trial_va], trial_s_va))

trial_gate_fit = Ridge(alpha=0.001).fit(
    trial_X, np.where(trial_relevant, 1.0, -1.0))
trial_gate_margins = trial_gate_fit.predict(trial_X)
print("Glucose gate ranges:",
      np.quantile(trial_gate_margins[trial_relevant], [0, .5, 1]),
      np.quantile(trial_gate_margins[~trial_relevant], [0, .5, 1]))

if trial_gate_result[0] <= 0:
    print("REJECTED locally: gating does not predict held-out innovations.")
else:
    trial_full_decoder = Ridge(alpha=trial_best_result[0]).fit(
        trial_X[trial_active], trial_y[trial_active])
    trial_full_classifier = LogisticRegression(C=1.0, max_iter=1000).fit(
        trial_X[trial_relevant], trial_active[trial_relevant])
    trial_P = np.asarray(rec_reg_projector, dtype=np.float64)
    trial_dv = trial_P @ trial_full_decoder.coef_
    trial_db = float(trial_full_decoder.intercept_)
    trial_av = trial_P @ trial_full_classifier.coef_[0]
    trial_ab = float(trial_full_classifier.intercept_[0])
    trial_gv = trial_P @ trial_gate_fit.coef_
    trial_gb = float(trial_gate_fit.intercept_)
    trial_B, trial_threshold = trial_gate_result[1:3]
    trial_gate_strength = 100.0

    trial_gain = np.asarray(wr_gain, dtype=np.float64)
    trial_gain_ref = float(np.median(trial_gain[trial_active]))
    trial_causal_beta = float(dml_results["insulin"]["all active"][0])
    trial_tilt_scale = trial_causal_beta / trial_gain_ref
    trial_bin_score = np.asarray(wr_score, dtype=np.float64)
    trial_bin_score -= trial_bin_score.mean()

    trial_W1_new = np.asarray(load_original("attr_heads.value_bin.0.weight")).copy()
    trial_b1_new = np.asarray(load_original("attr_heads.value_bin.0.bias")).copy()
    trial_W2_new = np.asarray(load_original("attr_heads.value_bin.2.weight")).copy()
    trial_b2_new = np.asarray(load_original("attr_heads.value_bin.2.bias")).copy()
    trial_neurons = trial_spare[:2]
    trial_b2_new += trial_W2_new[:, trial_neurons] @ trial_phi[:, trial_neurons].mean(axis=0)

    for neuron, sign in zip(trial_neurons, [1.0, -1.0]):
        trial_W1_new[neuron] = (
            trial_B * trial_av + trial_gate_strength * trial_gv
            + sign * trial_eps * trial_dv)
        trial_b1_new[neuron] = (
            trial_B * (trial_ab - trial_threshold)
            + trial_gate_strength * (trial_gb - 1.0)
            + sign * trial_eps * trial_db)
        trial_W2_new[:, neuron] = (
            sign * trial_tilt_scale * trial_bin_score / (2.0 * trial_eps))

    for name, value in [
        ("attr_heads.value_bin.0.weight", trial_W1_new),
        ("attr_heads.value_bin.0.bias", trial_b1_new),
        ("attr_heads.value_bin.2.weight", trial_W2_new),
        ("attr_heads.value_bin.2.bias", trial_b2_new),
    ]:
        save_edit(name, torch.as_tensor(value, dtype=torch.float32))

    print("Installed provisional insulin correction.")
    print("Replaced neurons:", trial_neurons.tolist())
    print("Causal slope / reference mean-response gain / tilt scale:",
          trial_causal_beta, trial_gain_ref, trial_tilt_scale)
    print("Current edits:", edited())
    with open("/work/notes.md", "a") as f:
        f.write(
            "\n## Run 53: first gated head trial\n"
            f"Active insulin innovation decoder held-out R2={trial_best_result[1]:.4f}, "
            f"but ungated inactive RMS={trial_best_result[2]:.3f}. "
            f"Infusion-gated all-hour R2={trial_gate_result[0]:.4f}; "
            f"inactive RMS={trial_gate_result[3]:.3f}. "
            f"Replaced two lowest-contribution value-head neurons {trial_neurons.tolist()} "
            f"with a glucose/active-insulin gated linear innovation correction; "
            f"slope={trial_causal_beta:.4f}, tilt={trial_tilt_scale:.5f}. "
            "Four head tensors provisionally edited. Must validate actual factual NLL "
            "and dose rollouts before retaining; no model gradients used.\n"
        )
```
