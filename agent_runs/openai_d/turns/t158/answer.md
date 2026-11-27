A modest norepinephrine correction costs little factual loss locally; some directions even improve it. I’ll test the \(10^{-5}\)-ridge direction at full probe calibration, whose overall local NLL increase is 0.00327. This remains provisional: only GPU factual predictions and actual dose rollouts can confirm it.

```python
import hashlib

r139_ridge = 1e-5
r139_target = 0.329011
r139_feature = r138_features[r139_ridge]
r139_scale = r139_target / r139_feature["probe_derivative"]
r139_candidate = {n: t.clone() for n, t in r137_W.items()}
r139_jplus, r139_jminus = map(int, r138_slots)
r139_v = torch.as_tensor(r139_feature["v"], dtype=torch.float32)
r139_b = float(r139_feature["b"])
r139_gv = torch.as_tensor(r132_gv, dtype=torch.float32)
r139_tilt = torch.as_tensor(r138_tilt, dtype=torch.float32)

r139_candidate["attr_heads.value_bin.0.weight"][r139_jplus] = (
    2*r139_gv + r138_alpha*r139_v)
r139_candidate["attr_heads.value_bin.0.weight"][r139_jminus] = (
    2*r139_gv - r138_alpha*r139_v)
r139_candidate["attr_heads.value_bin.0.bias"][r139_jplus] = (
    2*r132_gb + r138_alpha*r139_b)
r139_candidate["attr_heads.value_bin.0.bias"][r139_jminus] = (
    2*r132_gb - r138_alpha*r139_b)
r139_candidate["attr_heads.value_bin.2.bias"] += (
    r137_V[:, r138_slots] @ r137_hmean[r138_slots])
r139_candidate["attr_heads.value_bin.2.weight"][:, r139_jplus] = (
    r139_scale/(2*r138_alpha)*r139_tilt)
r139_candidate["attr_heads.value_bin.2.weight"][:, r139_jminus] = (
    -r139_scale/(2*r138_alpha)*r139_tilt)

r139_actual_L = F.linear(
    F.gelu(F.linear(
        r136_X, r139_candidate["attr_heads.value_bin.0.weight"],
        r139_candidate["attr_heads.value_bin.0.bias"])),
    r139_candidate["attr_heads.value_bin.2.weight"],
    r139_candidate["attr_heads.value_bin.2.bias"]) + r136_transition
r139_expected_L = (r138_L0 + r139_scale
                  * r139_feature["pair"][..., None] * r138_tilt)
r139_error = np.max(np.abs(r139_actual_L.numpy()-r139_expected_L))
r139_loss = r138_nll(r139_actual_L.numpy().astype(np.float64))
print("maximum logit construction error:", r139_error)
print("local NLL delta all/fit/validation:",
      [(r139_loss-r138_original_loss)[m].mean()
       for m in (np.ones(len(meta96), bool), r137_fit, r137_val)])
assert r139_error < 0.003
for n139, t139 in r139_candidate.items():
    assert t139.shape == load_original(n139).shape and torch.isfinite(t139).all()

r139_fallback = {n: t.clone() for n, t in r137_W.items()}
for n139, t139 in r139_candidate.items():
    save_edit(n139, t139)
r139_hash = hashlib.sha256(b"".join(
    r139_candidate[n].numpy().tobytes()
    for n in sorted(r139_candidate))).hexdigest()[:12]
np.savez("/work/insulin_ne_candidate_v1.npz",
         **{n: t.numpy() for n, t in r139_candidate.items()})
print("installed trial:", r139_hash, "edited tensors:", edited())

with Path("/work/notes.md").open("a") as note139:
    note139.write("\n## Run 139: NE trial installed\n"
                  f"Four-head-tensor candidate {r139_hash}: retained strength-2 insulin; "
                  f"added paired-GELU NE correction in neurons {r138_slots.tolist()}, "
                  f"ridge {r139_ridge}, diagnostic probe calibration {r139_target}. "
                  f"Local NLL delta overall {(r139_loss-r138_original_loss).mean():.6f}. "
                  "Strength-2 insulin-only fallback retained in r139_fallback and r115_strength2. "
                  "Pending paired GPU factual and actual dose-edit validation; not yet accepted.\n")
```
