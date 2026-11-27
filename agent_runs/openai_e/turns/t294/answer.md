The earlier work provides a measured MAP direction, but it did not validate this nonlinear edit. I’ll build an anchored one-step calculation that includes the recurrent block. This will let me measure the prelude-to-MAP response while avoiding the known error in absolute recurrent reconstruction.

```python
import dis, numpy as np, torch

def compact_dis254(fn):
    lines = {}
    current = 0
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None:
            current = ins.starts_line
        lines.setdefault(current, []).append(ins.opname + " " + ins.argrepr)
    for ln, ops in lines.items():
        print(str(ln) + ": " + " | ".join(ops))

print("WEIGHTS AND ANCHOR ALIGNMENT")
a254 = W0["core.b_in.weight"]
a254 = a254.detach().cpu().numpy() if isinstance(a254, torch.Tensor) else np.asarray(a254)
print("W0 vs original/current",
      np.max(np.abs(a254 - np.asarray(load_original("core.b_in.weight")))),
      np.max(np.abs(a254 - np.asarray(load("core.b_in.weight")))))
print("state_df", state_df.shape, list(state_df.columns))
print(state_df.iloc[actual_rows45[:6]].to_string(index=False))
print("actual_rows45", actual_rows45, "src45", src45)
print("bin directions", {k: (tuple(v.shape), float(np.linalg.norm(np.asarray(v))))
                          for k, v in bin_directions46.items()})

print("\nPRELUDE HIDDEN")
compact_dis254(prelude_hidden93)
print("\nCORE LAYER")
print("defaults", core_layer.__defaults__)
compact_dis254(core_layer)
print("\nCORE ATTENTION")
print("args", core_attn.__code__.co_varnames[:core_attn.__code__.co_argcount],
      "defaults", core_attn.__defaults__)
compact_dis254(core_attn)

print("\nMEMORY/STEP INTERFACES")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__"):
        c = v.__code__
        if (any(s in k.lower() for s in ["memory", "step", "block"])
            or any(isinstance(s, str) and "core.blocks.0" in s for s in c.co_consts)):
            print(k, c.co_varnames[:c.co_argcount],
                  "constants", c.co_consts[:16], "names", c.co_names)
print("\nMEMORY ARRAY CANDIDATES")
for k, v in list(globals().items()):
    if isinstance(v, (np.ndarray, torch.Tensor)) and (
        "mem" in k.lower() or "age" in k.lower()
    ):
        print(k, tuple(v.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 254: anchored recurrent differential interfaces\n"
            "Recovered measured b_in MAP directions; nonlinear edit remains untested. "
            "Inspecting attention/memory and selected-state alignment for a differential "
            "recurrent calculation anchored at exact GPU pre-coda states. "
            "Joint236 unchanged.\nPython runs: 254; forwards: 29.\n")
```