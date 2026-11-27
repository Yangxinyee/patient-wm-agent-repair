The reading rows align correctly, but the reconstructed hidden state is far from the GPU state. I also found a bookkeeping mistake: `ne_trial103.npz` stores the NE-only delta, so subtracting the insulin baseline was incorrect; the installed weights have not changed. I’ll check which weights the reconstruction helpers use and recover the contrast-row mapping. Fixing those checks is necessary before fitting a new direction.

```python
import numpy as np, torch, dis

ne127 = trial127["delta"].astype(np.float32)
print("=== CORRECTED EDIT COMPOSITION ===")
print("NE-only norm:", float(np.linalg.norm(ne127)))
print("current minus baseline+.35 NE:",
      float(np.linalg.norm(cur127-base127-.35*ne127)))

print("\n=== HIDDEN-STATE RECONSTRUCTION ===")
hg128 = arr127(core_state)[:, 192:960].reshape(-1,4,192)
hl128 = arr127(hlocal121)
dd128 = hl128-hg128
print("GPU h mean/std:", float(hg128.mean()), float(hg128.std()))
print("local h mean/std:", float(hl128.mean()), float(hl128.std()))
print("difference mean/std:", float(dd128.mean()), float(dd128.std()))
print("difference after removing bundle-wide scalar:",
      float(np.sqrt(np.mean((dd128-dd128.mean(axis=(1,2), keepdims=True))**2))))
for nm128 in ("hnew46", "hcheck46", "capture46"):
    xx128 = globals().get(nm128)
    if isinstance(xx128, dict):
        xx128 = xx128.get("hnew")
    if isinstance(xx128, (np.ndarray, torch.Tensor)):
        if tuple(xx128.shape) == (len(core_rows),4,192):
            print(nm128, "RMSE versus all-bundle reconstruction:",
                  float(np.sqrt(np.mean((arr127(xx128)-
                                         arr127(hnew_all121)[core_rows])**2))))

print("\n=== HELPER WEIGHTS ===")
seen128 = set()
for nm128, ob128 in list(globals().items()):
    if isinstance(ob128, dict) and "core.b_in.weight" in ob128:
        if id(ob128) in seen128:
            continue
        seen128.add(id(ob128))
        vv128 = ob128["core.b_in.weight"]
        if isinstance(vv128, (np.ndarray, torch.Tensor)):
            print(nm128,
                  "b_in minus original norm:",
                  float(np.linalg.norm(arr127(vv128)-
                                       arr127(load_original("core.b_in.weight")))))

for nm128 in ("core_lin", "core_layer", "core_attn", "_mf121"):
    ff128 = globals().get(nm128)
    if callable(ff128):
        print("\nHELPER", nm128,
              "args:", ff128.__code__.co_varnames[:ff128.__code__.co_argcount])
        print(" | ".join(f"{i.opname}:{i.argrepr}"
                         for i in dis.get_instructions(ff128)
                         if i.opname not in
                         ("RESUME", "CACHE", "PUSH_NULL", "PRECALL"))[:3300])

print("\n=== CONTRAST ROW MAPPING ===")
for nm128, ob128 in list(globals().items()):
    if isinstance(ob128, (np.ndarray, torch.Tensor)):
        if ob128.ndim == 1 and len(ob128) == 1497:
            print(nm128, str(ob128.dtype),
                  "first five:", arr127(ob128)[:5].tolist())
for nm128 in ("Y81", "cf_meta", "drug_meta", "drug_rows", "action_rows46"):
    ob128 = globals().get(nm128)
    if isinstance(ob128, dict):
        print(nm128, [(str(k), getattr(v, "shape", None))
                      for k,v in ob128.items()])
    elif ob128 is not None:
        print(nm128, "shape:", getattr(ob128, "shape", None))
        if hasattr(ob128, "columns"):
            print("columns:", list(ob128.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 128: hidden-state and edit-composition audit\n"
            "Reading source alignment is exact. GPU/local h RMSE=6.11; output-normalization hid a major "
            "reconstruction discrepancy. Corrected archive interpretation: ne_trial103 delta is NE-only, "
            "not baseline+NE; installed weights were never changed by this bookkeeping error. "
            "Inspecting helper weight sources and contrast indexing. Python runs: 128; forwards: 15.\n")
```