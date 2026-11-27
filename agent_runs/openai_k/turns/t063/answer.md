The existing posterior directions are too weak and variable: even large changes do not reliably reproduce the estimated responses. I won’t edit that pathway on this evidence. Next I’ll test the full action pathway and extract value-head features; a dose signal there could support a more targeted correction, while its absence would rule out that approach.

```python
import torch.nn.functional as F
t57 = time.time()

def r57_features(out, h, indices):
    captures = []
    previous_mlp = globals()["mlp0"]
    def capture_mlp(x, prefix, *args, **kwargs):
        y = previous_mlp(x, prefix, *args, **kwargs)
        if prefix == "attr_heads.value_bin":
            captures.append((x.clone(), y.clone()))
        return y
    globals()["mlp0"] = capture_mlp
    try:
        means = r49_readouts(out, h, indices)
    finally:
        globals()["mlp0"] = previous_mlp
    features = []
    errors = {}
    for wc, raw in captures:
        first = F.linear(wc, W0["attr_heads.value_bin.0.weight"],
                         W0["attr_heads.value_bin.0.bias"])
        candidates = {"gelu": F.gelu(first), "silu": F.silu(first),
                      "relu": F.relu(first)}
        errors = {k: float((F.linear(v, W0["attr_heads.value_bin.2.weight"],
                                     W0["attr_heads.value_bin.2.bias"])
                            - raw).abs().max())
                  for k, v in candidates.items()}
        features.append(candidates[min(errors, key=errors.get)].mean(1))
    return means, torch.stack(features, dim=1), errors

base57, Hbase57, err57 = r57_features(o56, h56, idx56)
print("Head feature reconstruction:", err57, tuple(Hbase57.shape))
print("Probe drug counts:", (rr39_rates[idx56] > 0).sum(0))
cf57 = {}
ctx57 = rr41_aug_inputs[:rr41_n, :192][idx56]
for drug57, j57 in (("insulin", 0), ("norepinephrine", 1)):
    for factor57 in (0.0, 0.5, 2.0, 3.0):
        u57 = rr35_scaled_action_encoding(idx56, drug57, factor57)
        z57 = mlp0(torch.cat([ctx57, core_probe_e, u57], dim=-1),
                   "z_post")[:, :32]
        a57, b57 = repaired_core_local(
            core_probe_h, core_probe_e, u57, z57,
            core_probe_memory, core_probe_mask, core_probe_ages)
        out57, h57 = (a57, b57) if a57.ndim == 2 else (b57, a57)
        means57, hidden57, _ = r57_features(out57, h57, idx56)
        dr57 = rr39_rates[idx56, j57] * (factor57-1)
        active57 = np.abs(dr57) > 1e-9
        delta57 = (means57-base57).numpy()
        slopes57 = ((dr57[active57, None]*delta57[active57]).sum(0)
                    / np.square(dr57[active57]).sum())
        cf57[(drug57, factor57)] = {
            "means": means57, "hidden": hidden57,
            "delta_rate": dr57, "out": out57, "h": h57}
        print(drug57, factor57, "immediate slopes",
              np.round(slopes57, 4).tolist(),
              "hidden change RMS",
              round(float((hidden57-Hbase57).square().mean().sqrt()), 5))

print("=== FULL-STATE ARRAY CANDIDATES ===")
print([(k, tuple(v.shape)) for k, v in list(globals().items())
       if hasattr(v, "shape") and len(v.shape) >= 2
       and v.shape[0] == 4035 and not k.startswith(("rr39", "rr41", "r49"))][:45])
print("Seconds:", round(time.time()-t57, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 57: reject weak existing posterior directions\n"
            "- Existing latent directions fail to give stable immediate causal responses; "
            "no posterior edit made.\n"
            "- Testing complete action-to-state effects and exact value-head features "
            "for a targeted, preservation-constrained head edit.\n"
            "- Python runs: 57; forward passes: 4.\n")
```
