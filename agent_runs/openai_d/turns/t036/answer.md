We have aligned training states and cross-fitted dose residuals. I’ll check their coverage and the estimated drug-response kernels before fitting anything. The proposed correction will use dose deviations from the recorded prescribing pattern; it is promising only if the head can represent those deviations and they explain its prediction errors.

```python
print("=== RECORD-BASED EFFECT ANALYSIS ===")
notes_text = Path("/work/notes.md").read_text()
start = notes_text.find("## Run 4")
stop = notes_text.find("## Run 7", start)
print(notes_text[start:stop][:5800])

print("\n=== HEAD METADATA ===")
print(head_meta.columns.tolist())
print(head_meta.head(5).to_string(index=False))
print("\n=== SAVED STATE ARRAYS ===")
for key, val in states0.items():
    print(key, np.asarray(val).shape, np.asarray(val).dtype)
print("First stay state object:",
      type(state_by_stay[next(iter(state_by_stay))]).__name__)
obj = state_by_stay[next(iter(state_by_stay))]
if isinstance(obj, dict):
    print({k: np.asarray(v).shape for k, v in obj.items()})
elif isinstance(obj, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in obj])
else:
    print(str(obj)[:300])

print("\n=== RESPONSE TARGETS ===")
for key, val in record_targets.items():
    print(key, str(val)[:1100])

print("\n=== LATENT / EMBEDDING TENSORS ===")
for nm in NAMES:
    if any(k in nm for k in ["z_prior", "z_post", "z_head", "gap", "embed.code",
                             "embed.kind", "embed.ekind", "embed.categorical"]):
        print(nm, SHAPES[nm])

print("\n=== COMPACT GLOBAL ARRAY / DICTIONARY NAMES ===")
print([(nm, type(globals()[nm]).__name__)
       for nm in sorted(list(globals()))
       if not nm.startswith("_")
       and isinstance(globals()[nm], (dict, pd.DataFrame, np.ndarray, torch.Tensor))
       and not nm.startswith(("rc_", "rh_"))][:90])
```
