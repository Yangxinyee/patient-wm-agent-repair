The earlier construction was intended to make innovations zero on inactive bundles. If that still holds, the likely leak is in attention transport rather than dose encoding. I’ll measure both signals across drug and non-drug bundles; nonzero transported signals with zero innovations would confirm that diagnosis.

```python
import torch.nn.functional as F

with torch.no_grad():
    x237 = torch.as_tensor(x236, dtype=torch.float32)
    zw237 = {
        k: torch.as_tensor(v, dtype=torch.float32)
        for k, v in safe234.items() if k.startswith("z_post.")
    }
    hz237 = F.gelu(F.linear(x237, zw237["z_post.0.weight"], zw237["z_post.0.bias"]))
    zp237 = F.linear(hz237, zw237["z_post.2.weight"], zw237["z_post.2.bias"])[:, :32]
    innovation237 = zp237[:, private_indices209].numpy()

    af237 = array196(att_features220(
        tokens222, route228["core.prelude.0.attn.q.weight"],
        route228["core.prelude.0.attn.kv.weight"]))
    signal237 = af237[..., response_cols228]
    if signal237.ndim != 3:
        raise RuntimeError(("unexpected signal shape", signal237.shape))

inactive237 = np.linalg.norm(x236[:, 384:], axis=1) < 1e-8
print("PRIVATE INNOVATIONS:")
for label, mask in (("inactive", inactive237), ("active", ~inactive237)):
    vals = innovation237[mask]
    print(label, "n", int(mask.sum()), "mean", vals.mean(0),
          "RMS", np.sqrt(np.mean(vals**2, axis=0)),
          "max_abs", np.abs(vals).max(0))

src237 = np.asarray(source201, dtype=int)
print("\nTRANSPORTED SIGNAL:", signal237.shape)
for label, mask in (("inactive", inactive237[src237]), ("active", ~inactive237[src237])):
    vals = signal237[mask].reshape(-1, 2)
    print(label, "n_bundles", int(mask.sum()), "mean", vals.mean(0),
          "RMS", np.sqrt(np.mean(vals**2, axis=0)),
          "centred_RMS", vals.std(0),
          "quantiles", np.quantile(vals, [0, .01, .5, .99, 1], axis=0))

print("\nINNOVATION-TO-SIGNAL LINEAR FIT:")
A237 = np.column_stack([innovation237[src237], np.ones(len(src237))])
B237 = signal237.mean(axis=1)
coef237 = np.linalg.lstsq(A237, B237, rcond=None)[0]
print("coefficients [insulin, NE, constant]:\n", coef237)
print("residual RMS", np.sqrt(np.mean((B237 - A237 @ coef237)**2, axis=0)))

print("\nATTENTION TENSOR NAMES:")
for name in NAMES:
    if name.startswith("core.prelude.0") and any(s in name for s in ("norm", "attn")):
        print(name, SHAPES[name])
print("\nATTENTION HELPER CONSTANTS:")
print(att_features220.__code__.co_consts)
print("helper locals", att_features220.__code__.co_varnames)
print("source mapping range", int(src237.min()), int(src237.max()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 237 — test attention leakage\n"
            "- Inspecting private innovations and reserved attention-value features separately on all retained bundles.\n"
            "- Background transport, not accumulation alone, is the leading explanation for response233's pre-treatment damage.\n"
            "- Neutral209 remains installed; no response weights changed.\n")
```