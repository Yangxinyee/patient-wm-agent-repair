The prior directions have the expected drug-specific signs, though MAP sensitivity is weaker than the record estimate. I’ll build a small dose-innovation carrier and measure its factual cost before adding a correction. A recoverable dose signal with little NLL change would justify the next step.

```python
import time
t80 = time.time()
torch.set_num_threads(8)

print("ACTION / COUNTERFACTUAL HELPERS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__"):
        if (any(s in name.lower() for s in ("action", "counter", "embed", "dose", "encode", "cf"))
                or ("rr33_u_all" in obj.__code__.co_names and name != "r59_core")):
            print(name, inspect.signature(obj), obj.__code__.co_names[:14])

idx80 = np.flatnonzero(valid76)
lookup80 = np.full(len(rr33_stays), -1, dtype=int)
lookup80[idx80] = np.arange(len(idx80))

post80_weights = {k: torch.as_tensor(v, dtype=torch.float32).clone()
                  for k, v in post66_base_weights.items()}
coef80 = arr71(post66_rate_models[0.1])
post80_weights["z_post.2.weight"][:32] += torch.tensor(
    arr71(carrier74) @ coef80[:-1].T, dtype=torch.float32)
post80_weights["z_post.2.bias"][:32] += torch.tensor(
    arr71(carrier74) @ coef80[-1], dtype=torch.float32)

def dataset80(indices, u=None, batch_size=96):
    hidden_parts, mean_parts, prob_parts = [], [], []
    for start in range(0, len(indices), batch_size):
        ix = np.asarray(indices[start:start+batch_size])
        ui = None if u is None else u[start:start+batch_size]
        out, h = r59_core(ix, u=ui)
        capture74.clear()
        features = r59_features(out, h, ix)
        if not capture74:
            raise RuntimeError("No prior context captured")
        ctx = capture74[-1]
        hidden = F.gelu(F.linear(ctx, W0["z_prior.0.weight"], W0["z_prior.0.bias"]))
        hidden_parts.append(np.c_[arr71(hidden), np.ones(len(ix))])
        mean_parts.append(arr71(features[0]))
        prob_parts.append(arr71(features[3]).mean(axis=2))
    return (np.concatenate(hidden_parts), np.concatenate(mean_parts),
            np.concatenate(prob_parts))

saved_weights80 = {k: W0[k] for k in post80_weights}
saved_mlp80 = mlp0
try:
    mlp0 = traced_mlp74
    P0_80, M0_80, probs0_80 = dataset80(idx80)
    for k, v in post80_weights.items():
        W0[k] = v
    P1_80, M1_80, probs1_80 = dataset80(idx80)
finally:
    for k, v in saved_weights80.items():
        W0[k] = v
    mlp0 = saved_mlp80

keep80 = lookup80[np.asarray(fidx59)] >= 0
rows80 = lookup80[np.asarray(fidx59)[keep80]]
bins80 = np.asarray(bins59)[keep80]
mask80 = np.asarray(observed59)[keep80].astype(bool)
truth0_80 = np.take_along_axis(probs0_80[rows80], bins80[...,None], axis=-1)[...,0]
truth1_80 = np.take_along_axis(probs1_80[rows80], bins80[...,None], axis=-1)[...,0]
dNLL80 = float(np.mean(np.log(np.maximum(truth0_80[mask80], 1e-12)) -
                      np.log(np.maximum(truth1_80[mask80], 1e-12))))
print("\nCARRIER LOCAL DIAGNOSTICS")
print("prior features:", P0_80.shape, P1_80.shape)
print("local factual dNLL:", dNLL80, "readings:", int(mask80.sum()))
print("mean-shift RMS by sensor:", np.sqrt(np.mean((M1_80-M0_80)**2, axis=0)))
print("prior-feature change RMS:", np.sqrt(np.mean((P1_80-P0_80)**2)))
print("seconds:", round(time.time()-t80, 1), "submitted edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 80: explicit carrier factual assay\n"
            f"- Built boundary-safe original/carrier prior datasets on {len(idx80)} bundles.\n"
            f"- Carrier local factual dNLL: {dNLL80:.7f}; no submitted edits.\n"
            "- All scratch weights restored. Forward passes: 4; Python runs: 80.\n")
```