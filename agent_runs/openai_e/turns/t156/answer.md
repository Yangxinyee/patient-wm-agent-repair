The coda replay matches the GPU hidden states to about two millionths and the output even more closely. That gives us a reliable anchor for immediate perturbation tests, though not a validated rollout. I’ll recover the reading-query inputs and corrected dose contrasts next; consistent decoding will let me fit an edit that limits factual-output changes.

```python
import numpy as np, torch, dis
from pathlib import Path

f135 = readout125
print("Reading helper arguments:", f135.__code__.co_varnames[:f135.__code__.co_argcount])
print("Reading helper locals:", f135.__code__.co_varnames)
print("Reading helper globals:", f135.__code__.co_names)
print("Reading helper constants:",
      tuple(x for x in f135.__code__.co_consts
            if isinstance(x, (str, int, float, tuple))))
print("Reading helper operations:",
      " ".join(f"{i.opname}:{i.argrepr}" for i in dis.Bytecode(f135)
               if i.opname in {"LOAD_GLOBAL", "LOAD_ATTR", "LOAD_CONST",
                               "STORE_FAST", "RETURN_VALUE"})[:4000])

print("\nReading-query arrays:")
for n135, v135 in list(globals().items()):
    if isinstance(v135, (np.ndarray, torch.Tensor)) and (
        (v135.ndim >= 1 and v135.shape[0] == 4751 and
         (v135.ndim > 1 or any(s in n135.lower()
                              for s in ["code", "bin", "src", "target"])))
        or any(s in n135.lower() for s in ["transition", "centres", "centers"])
    ):
        print(n135, tuple(v135.shape))

print("\nCounterfactual/guard objects:")
for n135, v135 in list(globals().items()):
    if any(s in n135.lower() for s in ["cf", "contrast", "guard"]) and (
        n135.endswith(("93", "94", "95", "100", "102", "103", "115"))
        or "meta" in n135.lower()
    ):
        if isinstance(v135, (np.ndarray, torch.Tensor)):
            print(n135, tuple(v135.shape))
        elif isinstance(v135, dict):
            print(n135, "dict", [(k, tuple(v.shape) if hasattr(v, "shape")
                                  else type(v).__name__)
                                 for k, v in list(v135.items())[:8]])
        elif hasattr(v135, "columns"):
            print(n135, tuple(v135.shape), list(v135.columns))
        elif isinstance(v135, (list, tuple)):
            print(n135, type(v135).__name__, len(v135))

print("\nDirection-basis shape/norms:", tuple(Dout120.shape),
      np.round(np.linalg.norm(np.asarray(Dout120), axis=0), 4))
notes135 = Path("/work/notes.md").read_text()
print("\nRecord-derived targets and accepted-edit evidence:")
evidence135 = [line for line in notes135.splitlines()
               if any(s in line.lower() for s in
                      ["target", "causal", "baseline77", "nll increase",
                       "doubling", "halving", "beta="])]
print("\n".join(evidence135[:30])[:3500])

with open("/work/notes.md", "a") as f135:
    f135.write("\n## Run 135: recover output-regression inputs\n"
               "Coda inversion is numerically exact (h RMSE 1.91e-6), avoiding the "
               "unvalidated recurrent reconstruction. Recovering reading queries, "
               "corrected counterfactual features and record-derived targets for "
               "a distribution-constrained closed-form edit. No weights changed. "
               "Python runs: 135; forwards: 15.\n")
```