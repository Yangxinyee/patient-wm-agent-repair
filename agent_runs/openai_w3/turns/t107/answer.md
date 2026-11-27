The remaining uncertainty is how a state perturbation translates into the model’s reading predictions. I’ll reconstruct that readout against existing factual outputs before calibrating the edit. Close agreement would make local sensitivity tests useful; disagreement would mean I should rely on GPU forward validation instead.

```python
import pandas as pd
print("FORWARD FILES")
for p96 in sorted(Path("/work").glob("fwd_*")):
    for f96 in sorted(p96.iterdir()):
        if f96.suffix in (".csv", ".npz"):
            if f96.suffix == ".csv":
                d96 = pd.read_csv(f96)
                print(str(f96), d96.shape, list(d96.columns))
            else:
                with np.load(f96) as z96:
                    print(str(f96), {k: z96[k].shape for k in z96.files})

print("\nVOCAB METADATA")
print("keys", list(VOCAB))
for k96 in ("config", "codes", "types", "event_kinds", "atom_kinds", "normalisation"):
    if k96 in VOCAB:
        print(k96, str(VOCAB[k96])[:1300])
print("CODE IDS", code_ids)
print("NORMALIZATION EXAMPLE", str(numeric_tables)[:1300])

print("\nCONTEXT HELPERS")
for n96 in ("context_with_flow", "state_context", "posterior_mean"):
    f96 = globals()[n96]
    print(n96, inspect.signature(f96))
    print(compact_bytecode(f96)[:1900])
print("LAST NUMERIC/CODE SUPPORT",
      [(n, SHAPES[n]) for n in NAMES if n.startswith(("gap", "embed.kind", "embed.ekind", "code_bias"))])
print("S SHAPES", {k: getattr(v, "shape", None) for k, v in S.items()})
print("PRELUDE FACTUAL FIT SD",
      [(str(k), np.round((as_numpy(hidden88).reshape(-1,768) @ v).std(0), 4).tolist())
       for k, v in contrast_fits88.items() if k in (1.0, (1.0,1e-8))])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 96; forwards 7. No edits. Matching local readout reconstruction "
            "to existing authoritative factual outputs before calibrating core injection.\n")
```