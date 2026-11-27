The scale-1 NE response remains within the factual budget on training replay (+0.00410 nats); larger strengths do not. I’ll submit scale 1 provisionally and test it on the GPU. It should make NE doubling raise MAP while leaving the insulin repair largely intact; either a factual-fit failure or the wrong rollout response would reject it.

```python
import pandas as pd
import json

backup_before195 = {
    name: as_np184(load(name)).copy() for name in edited()
}
selected195 = candidates194[1.0]
changed_names195 = sorted(set(edited()) | set(dual_candidate193) | {"core.b_in.weight"})
for name in changed_names195:
    save_edit(name, as_np184(selected195[name]).copy())
rW = dict(selected195)

paths195 = list(Path("/work/fwd_14").glob("*dose_edit*edited*.csv"))
if not paths195:
    paths195 = list(Path("/work/fwd_14").glob("*dose*edited*.csv"))
assert len(paths195) == 1, [str(p) for p in paths195]
rollout_table195 = pd.read_csv(paths195[0])
validation_ids195 = [int(x) for x in rollout_table195["stay_id"].drop_duplicates()]
assert 1 <= len(validation_ids195) <= 100

print("SAVED", edited())
print("VALIDATION STAYS", json.dumps(validation_ids195))
print("TRAINING REPLAY DELTA",
      results194[1.0]["delta_original"])

def compact195(value, depth=0):
    if isinstance(value, dict):
        return {str(k): compact195(v, depth+1)
                for k, v in list(value.items())[:10]}
    if isinstance(value, (np.ndarray, torch.Tensor)):
        a = as_np184(value)
        if a.size <= 30:
            return np.round(a, 6).tolist()
        return {"shape": list(a.shape)}
    if isinstance(value, (list, tuple)):
        if len(value) <= 20:
            return [compact195(v, depth+1) for v in value]
        return {"length": len(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)[:150]

print("\nRECORD-BASED KERNELS", compact195(kernels137))
print("RECORD-BASED READOUT ESTIMATES", compact195(readout_estimates137))
print("QUADRATIC FITS", compact195(quadratic137))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 195\n"
        "Saved provisional dual insulin/NE channel: amplitude 20, recalibrated "
        "NE decoder (.001,100), joint E ridge .0001, MAP-response direction "
        "ridge .1 at scale 1. Existing insulin response preserved by joint refit. "
        "Training replay dNLL +.0040994 versus original; scales 1.5/2 failed "
        "the budget. Previous submitted tensors backed up in backup_before195. "
        "GPU factual and rollout validation pending; causal readout targets "
        "remain record-based estimates subject to identification assumptions.\n"
    )
```