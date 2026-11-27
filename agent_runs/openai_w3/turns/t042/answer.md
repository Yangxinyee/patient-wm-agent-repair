The “isolated record” helper selects one event but does not ensure that its bundle contains only that event. Earlier tests may therefore have omitted companion readings. I’ll compare full-bundle reconstruction errors with genuinely single-atom bundles; accurate single-atom results would support that explanation.

```python
import numpy as np, torch

print("=== VOCAB CONVENTIONS ===")
for key in ("event_kinds", "atom_kinds", "numeric_note"):
    print(key, VOCAB.get(key))
print("numeric example:", VOCAB["numeric"][0] if isinstance(VOCAB["numeric"], list)
      else list(VOCAB["numeric"].items())[:1])

r37_idx = np.flatnonzero(same_stay)
r37_ctx = state_context(r37_idx)
r37_target = sx[r37_idx, 960:992]
r37_single = np.asarray(S["n_atoms"])[r37_idx] == 1
r37_drug = np.asarray(S["has_insulin"])[r37_idx] | np.asarray(S["has_norepinephrine"])[r37_idx]

print("\n=== FULL-BUNDLE POSTERIOR ERRORS ===")
r37_results = {}
for mode in ("preserve", "clear_numeric", "clear_all"):
    result37 = encode_bundles(end_mode=mode)
    print(mode, "return shapes:", [tuple(t.shape) for t in result37])
    r37_e, r37_u = result37[:2]
    r37_z = posterior_mean(r37_ctx, r37_e[r37_idx], r37_u[r37_idx])
    r37_err = ((r37_z - r37_target) ** 2).mean(dim=1).numpy()
    r37_results[mode] = (r37_e, r37_u, r37_err)
    for label, mask37 in (
        ("all", np.ones(len(r37_idx), dtype=bool)),
        ("single non-drug", r37_single & ~r37_drug),
        ("multiple non-drug", ~r37_single & ~r37_drug),
        ("single drug", r37_single & r37_drug),
        ("multiple drug", ~r37_single & r37_drug),
    ):
        if mask37.any():
            print(label, "n", int(mask37.sum()),
                  "MSE", float(r37_err[mask37].mean()),
                  "median", float(np.median(r37_err[mask37])))

print("\n=== TRUE SINGLE-ATOM RECORD EXAMPLES ===")
for i37 in r37_idx[r37_single & ~r37_drug][:8]:
    sid37, hour37 = int(S["stay_id"][i37]), float(S["hour"][i37])
    rows37 = state_records[
        (state_records.stay_id == sid37) &
        (np.abs(state_records.hour - hour37) < 0.0002)
    ]
    print("state", i37, "hour", hour37, "n_atoms", S["n_atoms"][i37],
          "records", rows37[["event", "value"]].to_dict("records"))

print("\n=== WEIGHT CACHE CHECK ===")
for name37 in ("encode.mlp1.0.weight", "encode.mlp2.0.weight",
               "encode.norm.weight", "embed.numeric.weight"):
    print(name37, float((W[name37] - load_original(name37)).abs().max()))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 37; forwards 3. isolated_record filters event, not bundle cardinality. Testing full sets and truly single-atom bundles separately to check whether omitted companion readings explain prior observation errors. No edits.\n")
```