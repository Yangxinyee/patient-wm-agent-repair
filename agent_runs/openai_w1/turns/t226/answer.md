The carrier barely changed the rollout effects, so this route is not useful as currently connected. I’ll restore the original weights and inspect the attention/state pathway. Any next test must demonstrate stronger transmission, rather than merely changing latent coordinates.

```python
carrier201_saved = {n: np200(load(n)) for n in edited()}
for n in list(edited()):
    drop_edit(n)

print("=== MODEL CONFIG ===")
print(VOCAB["config"])
print("\n=== PRELUDE / STATE TENSORS ===")
for n in NAMES:
    if re.search(r"prelude|b_in|c_out|coda|delta_in|a_in|out_norm|slots", n):
        print(n, SHAPES[n])

print("\n=== EXISTING CPU HELPERS ===")
for k in sorted(k for k,v in list(globals().items()) if callable(v)
                and re.search(r"att|core|step|replay|trans|norm|layer|coda|head|view|state",k,re.I)):
    print(k)

s201 = np.load("/work/fwd_14/states_original.npz")
print("\n=== ORIGINAL STATE EXPORT ===")
for k in s201.files:
    a = s201[k]
    print(k, a.shape, a.dtype)
    if a.dtype.kind == "f" and a.ndim >= 2:
        print(" RMS", float(np.sqrt(np.mean(a.astype(np.float64)**2))),
              "range", float(np.min(a)), float(np.max(a)))

print("\n=== CACHED CORE INPUT ARRAYS ===")
for k in sorted(k for k in list(globals())
                if re.match(r"^(ds_|r168_|rr_)",k))[:55]:
    v = globals()[k]
    if isinstance(v,(np.ndarray,torch.Tensor)):
        print(k, tuple(v.shape))
    elif isinstance(v,dict):
        print(k, "dict",list(v)[:8])
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 17–18 / Run 201\nCarrier gains 12/6: factual NLL unchanged at displayed precision; rollout insulin x2 -1.814 vs -1.773, half .791 vs .816; NE x2 .122 vs .048, half -.311 vs -.481. Rejected as ineffective; preserved tensors in carrier201_saved and restored all originals. Investigating attention/state transmission rather than further latent gain amplification. Counts: 201 Python runs, 18 forwards.\n")
```