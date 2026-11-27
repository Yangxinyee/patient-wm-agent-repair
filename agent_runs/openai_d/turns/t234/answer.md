The tuning states are available. I’ll reconstruct each reading’s preceding state and last-value register, then verify the head’s binning and register conventions. Matching the GPU’s original predictions is the check that must pass before I use this cache to compare edits.

```python
r200_s = dict(np.load("/work/fwd_28/states_original.npz"))
r200_data = pd.read_csv("/data/tuning.csv")
r200_data = r200_data[r200_data.stay_id.isin(r199_ids)].copy()
print("=== VOCAB STRUCTURE ===")
print("keys", list(VOCAB.keys()))
for k in ["config", "codes", "value_bin_centres", "numeric"]:
    if k in VOCAB:
        v = VOCAB[k]
        print(k, repr(v)[:2200])
print("=== ORIGINAL CACHE CONVENTIONS ===")
for k in ["bins", "gaps", "flow_means"]:
    print("R85", k, repr(R85.get(k))[:400])
print("head_meta first rows")
print(head_meta.head(5).to_string(index=False))
print("last-register tensors")
for n in NAMES:
    if any(t in n for t in ["last_numeric", "last_status", "embed_transition", "type_emb"]):
        print(n, SHAPES[n])
print("=== CACHE BUILD FUNCTIONS ===")
for k in sorted(list(globals())):
    if callable(globals()[k]) and any(t in k.lower() for t in ["head", "register", "r85", "r98", "r124", "build_meta"]):
        try:
            print(k, inspect.signature(globals()[k]))
        except Exception:
            pass
print("=== RELEVANT NOTE EXCERPTS ===")
r200_notes = open("/work/notes.md").read()
for part in r200_notes.split("\n## "):
    if any(t in part.lower() for t in ["register", "bin side", "last_status", "run 85", "run 98", "run 124"]):
        print(part[:900])
        if sum(len(p) for p in []) > 0:
            break
print("=== TUNING RECORD CHECK ===")
print(r200_data.head(8).to_string(index=False))
print("measurement count", int(r200_data.event.isin(
    ["glucose_finger","glucose_lab","map_arterial","map_cuff"]).sum()),
      "GPU count", len(r199_gpu))
```
