The output was truncated before the decisive comparisons. The projection identity is still the only firm finding. I’ll rerun a bounded check without printing large tensor defaults; agreement with GPU predictions, rather than either helper alone, will determine the correct assembly.

```python
print("INPUT COMPARISONS")
projection173 = (register168 @ W0["last_numeric.weight"].T).numpy()
for name in ("base162", "d169", "resid169"):
    obj173 = globals().get(name)
    if obj173 is not None:
        a173 = obj173.numpy() if torch.is_tensor(obj173) else np.asarray(obj173)
        if a173.shape == tuple(static168.shape):
            print(name, {
                "rms": float(np.sqrt(np.mean(a173**2))),
                "minus_static": float(np.sqrt(np.mean((a173-static168.numpy())**2))),
                "minus_projection": float(np.sqrt(np.mean((a173-projection173)**2))),
                "plus_projection": float(np.sqrt(np.mean((a173+projection173)**2)))
            })

print("\nSTATIC OBJECT NAMES")
for k, v in sorted(list(globals().items())):
    if re.search(r"static|with_code|reg0|query0|p0|q0|logits|probs", k, re.I):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            print(k, tuple(v.shape))
        elif callable(v):
            print(k, "function")
print("\nHELPER IMPLEMENTATIONS")
for name, limit in (("head_calc", 2200), ("readout125", 3300)):
    fn173 = globals().get(name)
    if fn173 is not None:
        print(name, "args", fn173.__code__.co_varnames[:fn173.__code__.co_argcount])
        print(compact_bytecode166(fn173, budget=limit))
print("\nGPU COLUMNS", list(gpu168.columns))
print("RECENT NLL", float(np.mean(nll168)),
      "GPU", float(gpu168["nll"].mean()) if "nll" in gpu168 else "see columns")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 173: bounded reference audit\n"
            "Run172 output truncated due tensor-valued function defaults. Repeating only decisive array "
            "comparisons and compact head/readout code. No weights changed. Python runs: 173; forwards: 22.\n")
```