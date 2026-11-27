The corrected decoder now matches the trusted helper exactly on identical inputs. I’ll check its accuracy against GPU predictions and recover the saved sensitivity directions and record-based effect estimates. Small GPU errors would support using it to design a constrained edit; large errors would make that approach unreliable.

```python
from pathlib import Path
import numpy as np, torch, re, dis

gpu126 = head_meta["gpu_mean"].to_numpy(dtype=float)
mlocal126 = readout125(hlocal121, olocal121)[0].numpy()
print("=== CORRECTED DECODER ACCURACY ===")
for ev126 in ["ALL"] + sorted(head_meta["event"].unique().tolist()):
    mask126 = (np.ones(len(head_meta), dtype=bool) if ev126 == "ALL"
               else head_meta["event"].to_numpy() == ev126)
    print(ev126, "n", int(mask126.sum()),
          "original-state RMSE", float(np.sqrt(np.mean(
              (mstate125[mask126]-gpu126[mask126])**2))),
          "reconstructed-state RMSE", float(np.sqrt(np.mean(
              (mlocal126[mask126]-gpu126[mask126])**2))),
          "reconstruction-only RMSE", float(np.sqrt(np.mean(
              (mlocal126[mask126]-mstate125[mask126])**2))))

print("\n=== SENSITIVITY INPUTS ===")
for nm126, ob126 in list(globals().items()):
    if nm126.endswith(("119", "120", "121", "122")):
        if isinstance(ob126, (np.ndarray, torch.Tensor)):
            print(nm126, tuple(ob126.shape), str(ob126.dtype))
        elif isinstance(ob126, dict):
            print(nm126, "keys", list(ob126)[:16])
print("Dout120 direction norms:", np.linalg.norm(Dout120, axis=0))
print("core.delta_in shape:", tuple(W0["core.delta_in"].shape),
      "softplus range:",
      float(torch.nn.functional.softplus(W0["core.delta_in"]).min()),
      float(torch.nn.functional.softplus(W0["core.delta_in"]).max()))
print("coda46 bytecode:")
print(" | ".join(f"{i.opname}:{i.argrepr}" for i in dis.get_instructions(coda46)
                 if i.opname not in ("RESUME", "CACHE", "PUSH_NULL", "PRECALL")))

print("\n=== SAVED EDITS ===")
for fn126 in ("baseline77.npz", "ne_trial103.npz", "ne_reduced107.npz"):
    with np.load("/work/"+fn126) as ar126:
        print(fn126, [(k, ar126[k].shape) for k in ar126.files])

print("\n=== SELECTED EARLIER EVIDENCE ===")
text126 = Path("/work/notes.md").read_text()
sections126 = re.split(r"(?m)(?=^## )", text126)
selected126 = [
    s for s in sections126
    if (re.search(r"^## (?:Run |Runs )(?:2[0-9]|3[0-9]|7[0-9]|8[0-4])\b", s)
        and any(w in s.lower() for w in
                ("target", "causal", "estimate", "accepted", "baseline77",
                 "nll", "calibrat", "direction")))
]
for s126 in selected126[-10:]:
    print(s126[:480].strip())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 126: validate type-corrected readout\n"
            "Type-one generic decoder exactly matches trusted helper. Comparing GPU and reconstructed "
            "states by outcome and recovering sensitivity/causal calibration inputs. No weights changed. "
            "Python runs: 126; forwards: 15.\n")
```