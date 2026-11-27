The recovered caches show that the repair substantially changed the prelude’s raw activations, but this alone does not establish why rollouts fail. I’ll inspect the exact replay interface and record-derived coefficients next. If the replay can separate action response from readout feedback, we can test a targeted correction; otherwise a smaller edit around the original model is safer.

```python
import inspect, dis, re
from pathlib import Path
import numpy as np, torch, pandas as pd

print("ESSENTIAL HELPER INTERFACES")
for n in ["state_update82", "response_raw234", "response186", "response_original228",
          "prelude155", "parts254", "raw_hidden228", "latent206",
          "original_tokens278", "normalize267", "next_view105", "readout134"]:
    v = globals().get(n)
    if callable(v):
        c = v.__code__
        print(n, inspect.signature(v), "\nnames:", c.co_names,
              "\nconstants:", repr(c.co_consts)[:500])

print("\nSMALL RESPONSE WRAPPER BYTECODE")
for n in ["response_raw234", "response186", "response_original228"]:
    v = globals().get(n)
    if callable(v):
        instr = list(dis.get_instructions(v))
        if len(instr) <= 100:
            print(n)
            print("\n".join(f"{x.opname} {x.argrepr}" for x in instr))

print("\nESTIMATES / SELECTED CACHE STRUCTURE")
for n in ["effect171", "zero169", "indices278", "rates279", "post_orig278",
          "checkpoint273", "ss280", "rs280", "r233_base", "r233_indices",
          "U239", "T239", "tokens239", "changed_tokens254", "dose_tokens254"]:
    v = globals().get(n)
    if isinstance(v, dict):
        if n == "effect171":
            print(n, v)
        else:
            print(n, list(v.keys()))
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        a = np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v)
        print(n, a.shape, "sample", a.reshape(-1)[:8])
    elif v is not None:
        print(n, type(v).__name__, repr(v)[:250])
print("Dataframes:", [(n, v.shape) for n, v in list(globals().items())
                      if isinstance(v, pd.DataFrame)][:25])

print("\nIDENTIFICATION NOTES (bounded)")
selected = [s for s in Path("/work/notes.md").read_text().splitlines()
            if re.search(r"identif|autoregress|instrument|randomiz|true effect|effect.*estimat|persistence|rho|lag coefficient", s, re.I)]
print("\n".join(selected[:12])[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 283; forwards 24. Bounded recovery of exact replay interface, physiological coefficients and changed-dose caches. Prelude edits are substantial; off-record instability remains a hypothesis, not an established cause. No weights changed.\n")
```
