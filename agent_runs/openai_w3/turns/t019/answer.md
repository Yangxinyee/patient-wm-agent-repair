The kernel still has the dose innovations and posterior features needed for a residual-based edit. Such an edit might respond strongly to changed doses but weakly to ordinary factual variation. I’ll first check reconstruction errors by bundle type; accurate drug-start states would let us test that idea without relying on the uncertain END encoding.

```python
import inspect
import torch.nn.functional as F

print("=== EXISTING FUNCTION SOURCE ===")
for fn in (make_atom, encode_bundles, mlp):
    try:
        src = inspect.getsource(fn)
        print(src[:5000])
    except (OSError, TypeError):
        print(fn.__name__, "source unavailable; constants:", fn.__code__.co_consts[:20])

print("=== ENCODER CONFIGURATION ===")
print("candidate:", candidate)
print("encoder_checks:", encoder_checks)
print("end_checks:", end_checks)
print("first bundle metadata:", bundle_atom_info[:3])

with torch.no_grad():
    reconstructed_z = linear(post_hidden, "z_post.2")[:, :32]
posterior_error = ((reconstructed_z - sx[valid_idx, -32:])**2).mean(1).numpy()
kinds = []
for idx in valid_idx:
    key0 = (int(S["stay_id"][idx]), round(float(S["hour"][idx]), 4))
    ns = len(start_atoms.get(key0, []))
    ne = len(end_atoms.get(key0, []))
    kinds.append("start+end" if ns and ne else "start" if ns else "end" if ne else "other")
kinds = np.array(kinds)
print("=== POSTERIOR ERROR BY BUNDLE TYPE ===")
for kind0 in np.unique(kinds):
    err = posterior_error[kinds == kind0]
    print(kind0, "n", len(err), "mean", err.mean(), "median", np.median(err),
          "max", err.max())
print("=== CORE TENSOR SHAPES ===")
for n in NAMES:
    if n.startswith("core.") and not n.startswith(("core.flow", "core.out_norm")):
        print(n, SHAPES[n])
print("=== NORMALIZATION SUMMARY ===")
for tab0 in VOCAB["numeric"]:
    print(tab0["code"], tab0["attribute"], "median", tab0["median"], "iqr", tab0["iqr"])
```