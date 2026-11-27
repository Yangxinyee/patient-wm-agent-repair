The earlier input-gain probe preserved factual fit but barely improved the dose rollouts. The readout edit failed its GPU check. I’ll now compare plausible register layouts on the saved original states. A correct layout should reduce systematic mean and NLL errors, not merely improve one sensor.

```python
import inspect
import json
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
print("=== RECONSTRUCTION SIGNATURE ===")
print(inspect.signature(reconstruct_reading_head))

audit_head = reconstruct_reading_head(hm, repair_states["x"])
audit_numbers = audit_head["numbers"].clone()
print("Numeric input shape:", tuple(audit_numbers.shape))
print("First numeric inputs:", audit_numbers[:5].numpy())
print("Centre keys:", list(VOCAB["value_bin_centres"])[:8])

audit_centres = []
for event in hm.event:
    c = np.asarray(VOCAB["value_bin_centres"][event], dtype=np.float32)
    audit_centres.append(np.r_[0.0, c] if len(c) == 16 else c)
audit_centres = torch.tensor(np.asarray(audit_centres), dtype=torch.float32)
audit_targets = torch.tensor(hm.target_bin.to_numpy(), dtype=torch.long)
audit_scale = hm.scale.to_numpy()
audit_original_mean = hm.original_mean.to_numpy()
audit_original_nll = hm.original_nll.to_numpy()
audit_without_numeric = audit_head["inputs"] - F.linear(
    audit_numbers, wt("last_numeric.weight"), wt("last_numeric.bias")
)
audit_transition = wt("embed_transition.value_bin.weight")[
    torch.tensor(hm.code.to_numpy() * 17 + hm.last_bin.to_numpy(), dtype=torch.long)
]

layout_results = []
layout_cache = {}
for value_col in range(2, 8):
    for presence_col in range(2, 8):
        if value_col == presence_col:
            continue
        num = torch.zeros_like(audit_numbers)
        num[:, :2] = audit_numbers[:, :2]
        num[:, value_col] = torch.tensor(hm.zlast.to_numpy(), dtype=torch.float32).clamp(-10, 10)
        num[:, presence_col] = torch.tensor(hm.has_last.to_numpy(), dtype=torch.float32)
        inp = audit_without_numeric + F.linear(
            num, wt("last_numeric.weight"), wt("last_numeric.bias")
        )
        hid = F.gelu(F.linear(
            inp, wt("attr_heads.value_bin.0.weight"), wt("attr_heads.value_bin.0.bias")
        ))
        logits = F.linear(
            hid, wt("attr_heads.value_bin.2.weight"), wt("attr_heads.value_bin.2.bias")
        ) + audit_transition
        probs = logits.softmax(-1)
        means = (probs * audit_centres).sum(-1).numpy()
        nll = -probs[torch.arange(len(hm)), audit_targets].clamp_min(1e-12).log().numpy()
        mean_error = (means - audit_original_mean) / audit_scale
        layout_results.append({
            "value_col": value_col, "presence_col": presence_col,
            "scaled_mean_rmse": np.sqrt(np.mean(mean_error**2)),
            "scaled_mean_bias": np.mean(mean_error),
            "mean_nll": nll.mean(),
            "nll_rmse": np.sqrt(np.mean((nll - audit_original_nll)**2)),
        })
        layout_cache[(value_col, presence_col)] = (means, nll)

layout_results = pd.DataFrame(layout_results).sort_values("scaled_mean_rmse")
print("\n=== BEST REGISTER LAYOUTS ===")
print(layout_results.head(10).to_string(index=False))
print("GPU original NLL:", audit_original_nll.mean())

print("\n=== FORWARD 12 STAYS ===")
audit_gpu12 = pd.read_csv("/work/fwd_12/factual_original.csv")
audit_tuning_ids = sorted(audit_gpu12.stay_id.unique().tolist())
print(audit_tuning_ids)
print("\n=== MATCHING WEIGHT-SHAPED ARRAYS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and tuple(obj.shape) == (17, 192):
        array = obj.detach().cpu().numpy() if torch.is_tensor(obj) else obj
        delta = array - load_original("attr_heads.value_bin.2.weight")
        print(name, "norm", round(float(np.linalg.norm(array)), 5),
              "delta singular values", np.round(np.linalg.svd(delta, compute_uv=False)[:3], 5))
```
