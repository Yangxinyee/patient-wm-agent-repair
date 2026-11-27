The finite-difference entries are close to float32 rounding increments, so the large directions may reflect numerical noise rather than genuine sensitivity. I will inspect the readout and measure the reserved head’s factual payload before recalibrating with larger perturbations. Clear, stable changes under those perturbations would support a real pathway; continued near-zero responses would mean we need a different readout edit.

```python
import dis
print("=== HELPER BYTECODE ===")
for name175 in ["prelude155", "view171", "phys171", "readout134"]:
    fn175 = globals()[name175]
    print("\n", name175)
    if name175 == "prelude155":
        ins175 = list(dis.get_instructions(fn175))
        print([(i.opname,i.argrepr) for i in ins175[-28:]])
    else:
        print(dis.Bytecode(fn175).dis()[:2600])
print("\n=== ALIGNED ARRAYS / WEIGHT HELPERS ===")
for name175 in sorted(list(globals())):
    if name175.startswith("_"): continue
    v175 = globals()[name175]
    if isinstance(v175,(torch.Tensor,np.ndarray)) and (
        tuple(v175.shape) == (4213,7,192) or
        name175 in ["attr_static105","transition97","centres105","codes97","basis171","O171"]):
        print(name175, tuple(v175.shape))
for name175 in ["W","tensor153","linear","layer_norm","r66_block","r66_core","r66_coda","r66_update"]:
    if name175 in globals():
        v175=globals()[name175]
        print(name175, type(v175).__name__,
              inspect.signature(v175) if callable(v175) else "")
print("\n=== HEAD LAYER SHAPES ===")
for name175 in NAMES:
    if any(s in name175 for s in ["attr_heads.value_bin","z_head","core.z_in","core.c_out","z_prior"]):
        print(name175, SHAPES[name175])
print("\n=== ZERO-HEAD SINGULAR VALUES ===")
vzero175 = kv174[336:384].numpy().astype(np.float64)
print(np.linalg.svd(vzero175,compute_uv=False)[:12])
with open("/work/notes.md","a") as f:
    f.write("\nPython run 175; forwards 10. J171 is visibly float32-quantized at epsilon .02, so its inverse directions are not trusted. Inspecting readout alignment, head payload, and layer structure before larger-perturbation calibration. Current edits unchanged.\n")
```