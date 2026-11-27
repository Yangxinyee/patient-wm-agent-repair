The retained tests show that the insulin correction largely disappears after the next factual reading, while the record-derived effect lasts several hours. Generated-reading feedback is therefore carrying much of the rollout response. I’ll inspect the latent carriers for a dose-sensitive adjustment that could strengthen doubling without also magnifying halving; factual replay will determine whether that is viable.

```python
import torch
print("=== RECORD-DERIVED DIRECT KERNELS ===")
for drug, value in kernels137.items():
    if isinstance(value, dict):
        print(drug, {k: (np.round(v, 4).tolist()
              if isinstance(v, np.ndarray) and v.size <= 24 else repr(v)[:200])
              for k, v in value.items()})
    else:
        print(drug, np.round(np.asarray(value), 4))

print("\n=== CURRENT POSTERIOR CARRIER ROWS ===")
cur223 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in NAMES}
orig223 = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES}
a223 = as_np184(cur223["z_post.0.weight"])
aorig223 = as_np184(orig223["z_post.0.weight"])
rows223 = np.flatnonzero(np.linalg.norm(a223 - aorig223, axis=1) > 1e-6)
print("modified first-layer rows", rows223.tolist())
for row in rows223:
    print("row", int(row), "bias", float(cur223["z_post.0.bias"][row]),
          "segment norms", [round(float(np.linalg.norm(a223[row, j:j+192])), 4)
                            for j in [0, 192, 384]],
          "output connections", [(int(j), round(float(v), 5))
              for j, v in enumerate(as_np184(cur223["z_post.2.weight"])[:, row])
              if abs(v) > 1e-5][:12])
z2diff223 = as_np184(cur223["z_post.2.weight"] - orig223["z_post.2.weight"])
print("modified output rows", np.flatnonzero(np.linalg.norm(z2diff223, axis=1) > 1e-6).tolist())
for j in np.flatnonzero(np.linalg.norm(z2diff223, axis=1) > 1e-6):
    if j < 32:
        print("latent", int(j), "bias", float(cur223["z_post.2.bias"][j]),
              "nonzero input count", int(np.count_nonzero(
                  np.abs(as_np184(cur223["z_post.2.weight"])[j]) > 1e-6)))
print("z_in changed column norms",
      [(int(j), round(float(v), 4)) for j, v in enumerate(
          np.linalg.norm(as_np184(cur223["z_in.weight"] - orig223["z_in.weight"]), axis=0))
       if v > 1e-5])
print("z_in.bias edit norm", float(torch.linalg.vector_norm(
      cur223["z_in.bias"] - orig223["z_in.bias"])))

print("\n=== GATE-CONSTRUCTION METADATA ===")
for name, obj in list(globals().items()):
    if re.search(r"(185|186|189|190|191|192)$", name):
        if inspect.isfunction(obj):
            print(name, inspect.signature(obj), "globals", obj.__code__.co_names)
        elif isinstance(obj, (float, int, str)):
            print(name, repr(obj)[:100])
        elif isinstance(obj, dict) and any(x in name.lower() for x in ["gate", "decoder", "candidate"]):
            print(name, "keys", list(obj)[:8])

print("\n=== CURRENT B_IN DECOMPOSITION CHECK ===")
curb223 = as_np184(cur223["core.b_in.weight"])
neunit223 = np.outer(as_np184(ne_directions206[10.0]), as_np184(joint194["ne"]))
print("||B - preserved + NE_unit||",
      float(np.linalg.norm(curb223 - as_np184(preserved_b194) + neunit223)))
print("insulin correction norm", float(np.linalg.norm(as_np184(ins_diff206))))
print("NE correction norm", float(np.linalg.norm(neunit223)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 223\nRetained teacher-forced impulses decay rapidly after subsequent observed readings, "
            "unlike the multi-hour record-derived kernels. Inspecting isolated latent carriers for a factual-safe "
            "dose-dependent adjustment; generated-reading feedback remains a key limitation. No saved weights changed.\n")
```
