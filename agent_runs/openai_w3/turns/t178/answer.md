The candidate uses four posterior neurons to encode dose innovations and two latent carrier directions. Its transport still needs calibration. I’ll check the recurrent update and state indexing first: a matching replay would make finite-difference calibration credible, while a mismatch would mean we must validate the injection through the forward tool instead.

```python
import dis, inspect
import numpy as np
import torch

print("=== RECURRENT HELPER BYTECODE ===")
for name in ("state_update82", "memory82"):
    fn160 = globals()[name]
    print("\n", name, inspect.signature(fn160))
    print("locals:", fn160.__code__.co_varnames)
    for ins in dis.get_instructions(fn160):
        if ins.opname not in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL",
                              "COPY", "SWAP"):
            print(ins.opname, ins.argrepr)

print("\n=== LAYER AND TENSOR HELPER CONSTANTS ===")
for name in ("layer82", "attention82", "tensor153", "posterior153",
             "next_view105", "context_with_flow"):
    fn160 = globals()[name]
    print(name, inspect.signature(fn160))
    print("locals", fn160.__code__.co_varnames)
    print("names", fn160.__code__.co_names)
    print("constants", [
        c if not isinstance(c, str) else c[:100]
        for c in fn160.__code__.co_consts
        if isinstance(c, (str, int, float, tuple)) or c is None
    ])

print("\n=== ALIGNMENT VARIABLES ===")
for name in sorted(list(globals())):
    if (any(s in name.lower() for s in
            ("align", "bundle", "positions", "index", "indices", "state_file"))
        and name not in ("SHAPES", "NAMES")):
        x = globals()[name]
        if isinstance(x, (np.ndarray, torch.Tensor)):
            print(name, tuple(x.shape),
                  x.reshape(-1)[:5].tolist() if x.size != 0 and
                  len(x.shape) == 1 else "")
        elif hasattr(x, "columns"):
            print(name, x.shape, list(x.columns))
        elif isinstance(x, (list, tuple, dict)):
            print(name, type(x).__name__, len(x))
print("\n=== TRAIN STATE FILES ===")
from pathlib import Path
for k in range(1, 11):
    for p in Path(f"/work/fwd_{k}").glob("states*.npz"):
        with np.load(p) as a:
            print(str(p), {n: a[n].shape for n in a.files})
```
