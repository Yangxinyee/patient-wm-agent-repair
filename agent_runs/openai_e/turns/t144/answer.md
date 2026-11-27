The architecture confirms that time flow, the latent prior, and the query type all affect the reading head. The output was cut off before the full helper comparison, so the mismatch remains unresolved. I’ll print a compact comparison, including captured defaults; a difference there could explain the decoder error without changing any weights.

```python
import dis, types, numpy as np, torch

for nm124 in ("next_means45", "readout121"):
    fn124 = globals().get(nm124)
    if not isinstance(fn124, types.FunctionType):
        print(nm124, "NOT FOUND")
        continue
    print("\n=== COMPACT BYTECODE:", nm124, "===")
    print("args:", fn124.__code__.co_varnames[:fn124.__code__.co_argcount])
    print("constants:", fn124.__code__.co_consts)
    ins124 = [
        f"{i.opname}:{i.argrepr}"
        for i in dis.get_instructions(fn124)
        if i.opname not in ("RESUME", "CACHE", "PUSH_NULL", "PRECALL")
    ]
    print(" | ".join(ins124))
    if fn124.__defaults__:
        print("defaults:",
              [(tuple(x.shape), str(x.dtype))
               if isinstance(x, (np.ndarray, torch.Tensor)) else repr(x)
               for x in fn124.__defaults__])

print("\n=== RETAINED CONTEXT SHAPES ===")
for nm124, ob124 in list(globals().items()):
    if isinstance(ob124, (np.ndarray, torch.Tensor)):
        if (nm124.endswith(("45", "120", "121", "122"))
                and any(s in nm124.lower() for s in
                        ("hnew", "hrec", "out", "src", "dt", "static", "trans",
                         "centre", "sens", "jac", "dirs", "basis"))):
            print(nm124, tuple(ob124.shape), str(ob124.dtype))
for nm124 in ("state_df", "core_rows", "state_x", "core_hnew", "core_h_all",
              "hnew_all121", "out_all121", "h_all121"):
    ob124 = globals().get(nm124)
    if ob124 is not None:
        print(nm124, "shape:", getattr(ob124, "shape", None),
              "len:", len(ob124) if hasattr(ob124, "__len__") else None)
        if hasattr(ob124, "columns"):
            print("columns:", list(ob124.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 124: compact decoder comparison\n"
            "Run123 output was truncated after architecture; no new numerical evidence. "
            "Comparing complete decoder bytecode and captured defaults without large array reprs. "
            "No weights changed. Python runs: 124; forwards: 15.\n")
```