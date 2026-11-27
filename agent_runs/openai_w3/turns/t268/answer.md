The payload occupies two reserved attention channels, and factual compensation is fitted through the prelude FFN. I’ll verify the cached fitting arrays against the installed weights before testing lower insulin and higher norepinephrine gains. That test would show whether gain calibration can improve the rollouts or whether their asymmetric dynamics need a different repair.

```python
print("FITTING DATA AND PAIR METADATA")
for name240 in ["T154", "C154", "E154", "U154", "tokens206",
                "orig_raw220", "X220", "H220", "R220",
                "hm220", "hs220", "tr220", "va220", "cov_base220",
                "cov_contrast220", "ridge_diag220", "refit220",
                "contrast_features220", "pairs", "paired", "dose_values",
                "dose154", "rates154", "zero169"]:
    if name240 not in globals():
        continue
    v240 = globals()[name240]
    if isinstance(v240, dict):
        print(name240, {str(k): (tuple(v.shape) if hasattr(v,"shape")
                               else (type(v).__name__, len(v) if hasattr(v,"__len__") else ""))
                        for k,v in v240.items()})
    elif hasattr(v240, "shape"):
        print(name240, tuple(v240.shape))
    else:
        print(name240, type(v240).__name__)

print("\nOTHER ACTION-PAIR NAMES")
print(" ".join(sorted(k for k in dict(globals())
                     if any(s in k.lower() for s in
                            ["paired", "u154", "dose154", "rate154",
                             "physical", "185", "192", "214"]))))

print("\nCACHED FFN WEIGHT CHECK")
for name240, v240 in [("w2_220", w2_220), ("b2_220", b2_220)]:
    tn240 = "core.prelude.0.ffn.2." + ("weight" if name240.startswith("w") else "bias")
    a240 = as_numpy(v240)
    print(name240, "vs installed", float(np.linalg.norm(a240-as_numpy(load(tn240)))),
          "vs original", float(np.linalg.norm(a240-as_numpy(load_original(tn240)))))

parts240 = prelude155(tokens206, overrides=checkpoint222)
h240 = F.gelu(F.linear(layer_norm(parts240[1], "core.prelude.0.norm3"),
                       tensor153(load("core.prelude.0.ffn.0.weight")),
                       tensor153(load("core.prelude.0.ffn.0.bias"))))
raw240 = parts240[1] + F.linear(
    h240, tensor153(load("core.prelude.0.ffn.2.weight")),
    tensor153(load("core.prelude.0.ffn.2.bias")))
print("\nINSTALLED BASELINE RAW RMS VS ORIGINAL",
      float(np.sqrt(np.mean((arr183(raw240)-np.asarray(orig_raw220))**2))))
print("PAYLOAD COLUMNS NORMS",
      np.linalg.norm(as_numpy(load("core.prelude.0.attn.out.weight"))[:,[144,145]],axis=0))
print("X220 ranges", float(np.min(X220)), float(np.max(X220)),
      "last column", np.unique(np.asarray(X220)[:,-1])[:6])
print("split counts", len(tr220), len(va220))
with open("/work/notes.md","a") as f:
    f.write("\nPython run 240; forwards 16. Verifying cached fitting data against installed checkpoint222 before constructing reversible payload-gain tests. No weights changed.\n")
```