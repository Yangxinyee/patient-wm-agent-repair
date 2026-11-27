The architecture confirms that last-reading features feed a learned projection. I’ll compare the residual with those projection columns and inspect the helper’s indexing. An exact match would identify the assembly error; otherwise, I’ll keep calibration on hold.

```python
import dis, numpy as np

print("STATIC BUILDER")
print(compact_bytecode166(build_static167, budget=3600))
print("\nREGISTER PROJECTIONS")
for name in ("last_numeric.weight", "embed.numeric.weight"):
    A171 = W0[name].detach().cpu().numpy().astype(np.float64)
    C171 = np.linalg.lstsq(A171, coef169[1:].T, rcond=None)[0]
    print(name, A171.shape,
          "residual RMS", np.sqrt(np.mean((A171 @ C171 - coef169[1:].T)**2)))
    print("feature coefficients:\n", np.round(C171, 5))

print("\nRELEVANT OBJECTS")
for k, v in sorted(list(globals().items())):
    if re.search(r"(162)$", k) or k in (
        "static0", "reg0", "register0", "trans0", "centres0", "dt0",
        "state_x", "flow_h", "train_meta162", "meta162", "W0"
    ):
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:8] if isinstance(v, dict) and k != "W0" else "")
        if callable(v) and hasattr(v, "__code__"):
            print("  args:", inspect.signature(v), "names:", v.__code__.co_names)
print("DESIGN FIRST ROWS", design169[:4])
print("METADATA FIRST ROWS")
print(aligned_meta168.head(4).to_string(index=False))

evidence171 = [line for line in nt170.splitlines()
               if re.search(r"target|causal|RD |discontinu|baseline77|insulin.only", line, re.I)]
print("\nSELECTED LATER EVIDENCE")
for line in evidence171[-22:]:
    print(line[:420])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 171: locate register projection discrepancy\n"
            "Comparing the exactly register-linear static residual with learned numeric projection columns "
            "and inspecting builder indexing. No weights changed. Python runs: 171; forwards: 22.\n")
```