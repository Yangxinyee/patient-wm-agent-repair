The attention-output edit is about the same scale as the large transport directions, suggesting it contains the main payload. The failed candidate changed only the FFN correction. I’ll identify the no-payload checkpoint and fitting data so the next experiment can reduce the attention payload itself and refit factual compensation.

```python
print("PAYLOAD CHECKPOINTS")
for k248 in ("zero169", "base_override228", "checkpoint193", "checkpoint222", "post206"):
    v248 = globals().get(k248)
    if isinstance(v248, dict):
        print(k248, "keys", list(v248))
        for n248 in ("core.prelude.0.attn.out.weight", "core.prelude.0.ffn.2.weight", "core.prelude.0.ffn.2.bias"):
            if n248 in v248:
                print(" ", n248, "delta from installed",
                      float(np.linalg.norm(np.asarray(v248[n248])-np.asarray(load(n248)))),
                      "norm", float(np.linalg.norm(np.asarray(v248[n248]))))

print("\nREPLAY DATA")
for k248 in ("tokens206", "orig_raw220", "orig_Eflat220", "H220", "R220", "tr220", "va220",
             "C154", "E154", "U154", "zin154", "zb154", "h186", "out186", "r233_raw"):
    v248 = globals().get(k248)
    print(k248, type(v248).__name__, tuple(v248.shape) if hasattr(v248, "shape") else "")
print("PAIRED FEATURE OBJECTS")
for k248, v248 in sorted(list(globals().items())):
    if any(s248 in k248.lower() for s248 in ("pair", "contrast", "factors", "hidden", "tokens", "gain")):
        if isinstance(v248, dict):
            print(k248, "dict keys", list(v248)[:10])
        elif isinstance(v248, (np.ndarray, torch.Tensor)):
            print(k248, tuple(v248.shape))

print("\nPRELUDE OPERATIONS")
f248 = prelude155
for ins248 in dis.get_instructions(f248):
    if ins248.opname not in ("RESUME", "CACHE", "PUSH_NULL", "PRECALL", "LOAD_FAST",
                            "LOAD_CONST", "STORE_FAST", "COPY", "SWAP"):
        print(ins248.opname, ins248.argrepr)

with open("/work/notes.md", "a") as f248:
    f248.write("\nPython run 248; forwards 18. Attention-output delta matches the large payload scale; candidate243 altered only FFN. "
               "Retrieving no-payload checkpoint and fitting data to scale attention payload itself, with refitted factual compensation. "
               "No new edits.\n")
```