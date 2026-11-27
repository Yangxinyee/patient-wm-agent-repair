The fitting cache contains paired dose changes across several multipliers, but its factual coverage is only 100 stays. I’ll check the target semantics and weighting, then add the independent factual states to the analytic fit. If broader anchoring removes the loss spike without erasing dose sensitivity, the small fitting sample was the main problem.

```python
print("=== FITTER INSTRUCTIONS ===")
print("\n".join(f"{i.offset:4d} {i.opname:24s} {i.argrepr}"
                for i in dis.get_instructions(fit_arch215)))
print("\n=== FEATURE BUILDER, SECOND HALF ===")
print("\n".join(f"{i.offset:4d} {i.opname:24s} {i.argrepr}"
                for i in dis.get_instructions(make_arch215) if i.offset >= 560))

print("\n=== CANDIDATE FEATURE SHAPES ===")
for k, v in arch219.items():
    print(k, getattr(v, "shape", None),
          list(v.keys())[:3] if isinstance(v, dict) else
          (str(v)[:100] if not hasattr(v, "shape") else ""))
p233 = paired214[("insulin", 2.0)]
print("paired target means/std",
      [(k, float(np.mean(p233[k])), float(np.std(p233[k])))
       for k in ["lf", "li", "target", "dr"]])
print("Headbase original agreement",
      np.max(np.abs(headbase214[:-1].T -
                    np.asarray(load_original("attr_heads.value_bin.2.weight")))),
      np.max(np.abs(headbase214[-1] -
                    np.asarray(load_original("attr_heads.value_bin.2.bias")))))
print("DataFrames",
      [(k, tuple(v.shape)) for k, v in list(globals().items())
       if type(v).__name__ == "DataFrame" and re.search(r"(22[4-8]|179|tune)", k)])
```
