The linear MAP candidates offer only modest dose-response gains within the remaining loss budget. I’ll first test a small, record-fitted bin-calibration model for factual miscalibration. If it improves both training and tuning likelihood, it could fund a stronger causal correction; a training-only gain would not justify using it.

```python
import torch
import torch.nn.functional as F

def cal170_nll(mc, addition=None):
    logits = mc["logits"]
    if addition is not None:
        logits = logits + torch.as_tensor(addition, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[1]
    marginal = probs.reshape(-1, 4, n, 17).mean(1)
    target = mc["target"].long()
    return -marginal.gather(
        2, target[None, :, None].expand(marginal.shape[0], -1, 1)
    ).squeeze(-1).clamp_min(1e-30).log().mean(0)

cal170_base_train = cal170_nll(wm_train145).numpy()
cal170_base_tune = cal170_nll(wm_tune145).numpy()
cal170_train_events = rebuilt_train145.event.to_numpy()
cal170_tune_events = wm_tune_meta145.event.to_numpy()
cal170_biases = {}
cal170_rows = []
for alpha170 in [5.0, 20.0, 80.0]:
    bias170 = np.zeros((4510, 17), dtype=np.float32)
    tbias170 = np.zeros((4751, 17), dtype=np.float32)
    for sensor170 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        m170 = cal170_train_events == sensor170
        tm170 = cal170_tune_events == sensor170
        counts170 = np.bincount(wm_train145["target"].numpy()[m170], minlength=17)
        expected170 = wm_train145["probs"].mean(0).numpy()[m170].sum(0)
        b170 = np.log((counts170 + alpha170) / (expected170 + alpha170))
        b170 -= b170.mean()
        cal170_biases[(alpha170, sensor170)] = b170.astype(np.float32)
        bias170[m170] = b170
        tbias170[tm170] = b170
    for gain170 in [0.25, 0.5, 1.0]:
        dn170 = cal170_nll(wm_train145, gain170 * bias170).numpy() - cal170_base_train
        tdn170 = cal170_nll(wm_tune145, gain170 * tbias170).numpy() - cal170_base_tune
        cal170_rows.append({"alpha": alpha170, "gain": gain170,
                            "train_dNLL": dn170.mean(), "tuning_dNLL": tdn170.mean(),
                            "tuning_glucose": tdn170[np.isin(
                                cal170_tune_events, ["glucose_finger", "glucose_lab"])].mean(),
                            "tuning_MAP": tdn170[np.isin(
                                cal170_tune_events, ["map_arterial", "map_cuff"])].mean()})
print(pd.DataFrame(cal170_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n=== ABSENT-VALUE PROBABILITY ===")
for label170, mc170, ev170 in [
    ("train", wm_train145, cal170_train_events),
    ("tuning", wm_tune145, cal170_tune_events)
]:
    p0170 = mc170["probs"].mean(0).numpy()[:, 0]
    print(label170, pd.DataFrame({"event": ev170, "p0": p0170})
          .groupby("event").p0.mean().to_dict())
print("\n=== TRAIN SPLIT ===")
print("wm_split151:", np.unique(wm_split151, return_counts=True))
print("train stays:", rebuilt_train145.stay_id.nunique(),
      "tuning stays:", wm_tune_meta145.stay_id.nunique())
```
