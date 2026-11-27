The last run failed while printing the encoder’s return values, before testing the hypothesis. I’ll handle that return structure correctly and repeat the comparison. The key test remains whether complete bundles—or truly single-atom bundles—match the saved posterior states.

```python
import numpy as np, torch, dis

print("encode_bundles return construction:")
print(dis.Bytecode(encode_bundles).dis()[-1100:])

r38_idx = np.flatnonzero(same_stay)
r38_ctx = state_context(r38_idx)
r38_target = sx[r38_idx, 960:992]
r38_single = np.asarray(S["n_atoms"])[r38_idx] == 1
r38_drug = np.asarray(S["has_insulin"])[r38_idx] | np.asarray(S["has_norepinephrine"])[r38_idx]
r38_results = {}

for mode38 in ("preserve", "clear_numeric", "clear_all"):
    returned38 = encode_bundles(end_mode=mode38)
    parts38 = list(returned38.values()) if isinstance(returned38, dict) else list(returned38)
    print("\n", mode38, "parts:",
          [(type(t).__name__, getattr(t, "shape", None)) for t in parts38])
    tensors38 = [
        t for t in parts38
        if isinstance(t, torch.Tensor) and tuple(t.shape) == (len(sx), 192)
    ]
    if len(tensors38) < 2:
        print("No unambiguous pair of bundle tensors; skipping rather than assuming.")
        continue
    e38, u38 = tensors38[:2]
    z38 = posterior_mean(r38_ctx, e38[r38_idx], u38[r38_idx])
    err38 = ((z38 - r38_target) ** 2).mean(dim=1).numpy()
    r38_results[mode38] = (e38, u38, err38)
    for label38, mask38 in (
        ("all", np.ones(len(r38_idx), dtype=bool)),
        ("single non-drug", r38_single & ~r38_drug),
        ("multiple non-drug", ~r38_single & ~r38_drug),
        ("single drug", r38_single & r38_drug),
        ("multiple drug", ~r38_single & r38_drug),
    ):
        if mask38.any():
            print(label38, "n", int(mask38.sum()),
                  "MSE", float(err38[mask38].mean()),
                  "median", float(np.median(err38[mask38])))

print("\nTrue single-atom examples:")
for i38 in r38_idx[r38_single & ~r38_drug][:6]:
    rows38 = state_records[
        (state_records.stay_id == int(S["stay_id"][i38])) &
        (np.abs(state_records.hour - float(S["hour"][i38])) < 0.0002)
    ]
    print(i38, float(S["hour"][i38]), rows38[["event", "value"]].to_dict("records"))

print("\nOriginal cache differences:")
for name38 in ("encode.mlp1.0.weight", "encode.mlp2.0.weight",
               "encode.norm.weight", "embed.numeric.weight"):
    print(name38, float((W[name38] - load_original(name38)).abs().max()))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 38; forwards 3. Run 37 failed only on return-shape printing before comparisons. Repeating full-bundle and cardinality-stratified posterior tests with safe return handling. No edits.\n")
```