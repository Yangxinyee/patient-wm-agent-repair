The larger correction still barely affects glucose and moves the norepinephrine response in the wrong direction. Posterior correlations therefore do not identify a useful editing direction. I’ll revert this probe and inspect the recurrent computation so I can calibrate directions from actual forward responses, without gradients.

```python
candidate_large132 = {n: np.asarray(load(n)).copy() for n in edited()}
for n in list(edited()):
    drop_edit(n)
print("current edits after rejecting probe:", edited())

print("=== CORE TENSORS ===")
for n in NAMES:
    if n.startswith(("core.","z_in.","z_prior.")):
        print(n, SHAPES[n])

print("=== RECONSTRUCTION HELPERS ===")
for name in ("head_exact","head2","reconstruct_head","latent_context",
             "posterior109","pool109","drug_u99","complete_atom57"):
    fn = globals().get(name)
    if callable(fn):
        print(name, inspect.signature(fn))
        print(" names:", fn.__code__.co_names)
        print(" constants:", repr(fn.__code__.co_consts)[:450])

print("=== STATE FILES ===")
state_paths132 = sorted(Path("/work").glob("fwd_*/states_*.npz"))
for p in state_paths132:
    s = np.load(p)
    print(str(p), {k: tuple(s[k].shape) for k in s.files})
print("=== RETAINED STATE/HEAD OBJECTS ===")
for k,v in list(globals().items()):
    if k.startswith("_"):
        continue
    if (k.startswith("latent_") or k.endswith("104") or
        k in ("S","states","head_data","head_states","bundle_rows65","bundle_lookup65")):
        if hasattr(v,"shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v,dict):
            print(k, "dict", list(v)[:12])
        elif isinstance(v,(list,tuple)):
            print(k, type(v).__name__, len(v))
        else:
            print(k, type(v).__name__)

with open("/work/notes.md","a") as f:
    f.write("\n## Python 132; forwards 9–10 — reject correlation-based latent direction\n"
            "- At gains (-200,+150), factual NLL improved .00026 but insulin x2 only -1.884 and NE x2 +.027.\n"
            "- Larger gains fail to repair effects; NE moves opposite the record-based target.\n"
            "- Reverted all edits; saved probe weights in candidate_large132.\n"
            "- Next: calibrate posterior latent directions against actual recurrent/head forward responses, not posterior/reading correlations.\n")
```
