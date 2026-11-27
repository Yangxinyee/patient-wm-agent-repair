The provisional record-based rollout targets are roughly −14.2/+7.1 for insulin and +4.7/−2.3 for norepinephrine, well beyond the original model’s responses. The gate may have been fitted after a projection that was not applied consistently. I’ll test that directly and identify the inputs that reproduce the cached hidden features; failure would rule out installing those coefficients.

```python
import torch.nn.functional as F

def cpu_tensor(x):
    return x.detach().to(dtype=torch.float32, device="cpu") if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

q68_a = cpu_tensor(rep_Q68)
fixed68_a = cpu_tensor(rep_fixed68)
nf_a = cpu_tensor(wr_nf)
ln_a = F.linear(nf_a, load_original("last_numeric.weight"),
               load_original("last_numeric.bias"))
h68_a = cpu_tensor(rep_H68)
w1_a = load_original("attr_heads.value_bin.0.weight")
b1_a = load_original("attr_heads.value_bin.0.bias")

wc_candidates79 = {
    "Q68 + fixed68": q68_a + fixed68_a,
    "Q68 + fixed68 + numeric": q68_a + fixed68_a + ln_a,
    "Q68 + wr_fixed + numeric": q68_a + cpu_tensor(wr_fixed) + ln_a,
}
print("=== RAW-INPUT RECONSTRUCTION ===")
for label, x in wc_candidates79.items():
    h = F.gelu(F.linear(x, w1_a, b1_a))
    print(label, "hidden RMS", float(torch.sqrt(torch.mean((h-h68_a)**2))))

P79 = np.asarray(rec_reg_projector, dtype=np.float64)
v79 = np.asarray(rec_gate_v, dtype=np.float64)
print("\n=== PROJECTION AUDIT ===")
print("||P v - v||:", float(np.linalg.norm(P79 @ v79 - v79)))
print("raw mean projected versus stored feature RMS:",
      float(np.sqrt(np.mean((np.asarray(repair_wc_mean) @ P79
                             - np.asarray(rec_head_X))**2))))
for label, x in wc_candidates79.items():
    xm = x.mean(0).numpy()
    for variant, v in (("original v", v79), ("projected v", P79 @ v79)):
        scores = xm @ v + rec_gate_b
        print(label, variant)
        print(pd.DataFrame({"event": wr_meta.event, "score": scores})
              .groupby("event").score.agg(["min", "max", "mean"]).round(3))

print("\n=== REGISTER ARRAYS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and any(
        s in name.lower() for s in ("nf", "register", "reg68")
    ):
        print(name, tuple(obj.shape))
print("wr_nf rows 2..6:", nf_a[2:7])
print("\n=== LATEST RECONSTRUCTION VARIABLE NAMES ===")
print([n for n in globals() if n.startswith(("rep_", "draw", "cache"))])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 74–79: restart audit\n"
            "No edits. Recovered record-based immediate effects: insulin about -1.99 "
            "per rate unit; norepinephrine about +3.29. Provisional controlled-kernel "
            "rollout targets: insulin x2 -14.18, half +7.09; NE x2 +4.68, half -2.34. "
            "Stored reading gate is excellent on its fitted projected features but "
            "fails on sampled raw head inputs. Auditing feature projection and "
            "reconstruction before any new installation.\n")
```
