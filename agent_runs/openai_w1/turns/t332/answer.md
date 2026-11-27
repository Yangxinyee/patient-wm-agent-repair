The candidate failed the factual limit by 0.084 nats, with most damage in the MAP channels. Glucose loss changed much less. I’ll reject it, reduce the norepinephrine injection tenfold, and modestly increase the insulin injection. This tests whether a useful insulin response can fit within the overall loss budget; the revised candidate must still pass paired factual validation.

```python
import numpy as np, pandas as pd, re
from pathlib import Path

print("Run 290/300; forwards 31/50.")
rejected290 = {n: np.asarray(load(n)).copy() for n in edited()}
gain290 = np.array([-3.0, 0.4])
delta290 = (B289 * gain290[None, :]) @ P289.T
candidate290 = {**baseline289,
                bname289: np.asarray(load_original(bname289)) + delta290.astype(np.float32)}
for n, a in baseline289.items():
    save_edit(n, a)
save_edit(bname289, candidate290[bname289])
print("Installed revised gains", gain290.tolist(),
      "b_in delta RMS", float(np.sqrt(np.mean(delta290**2))))

o290 = pd.read_csv("/work/fwd_31/factual_original.csv")
e290 = pd.read_csv("/work/fwd_31/factual_edited.csv")
print("Forward31 overall NLL delta", float(e290.nll.mean()-o290.nll.mean()))
eventcol290 = next(c for c in ("event", "code", "event_code") if c in o290.columns)
print("Per-event original/candidate NLL")
print(pd.DataFrame({"original": o290.groupby(eventcol290).nll.mean(),
                    "candidate": e290.groupby(eventcol290).nll.mean()}).to_string())
print("Current baseline versus safe236 maximum differences",
      {n: float(np.max(np.abs(baseline289[n]-np.asarray(a))))
       for n, a in safe236.items() if n in baseline289})

print("Compact record-effect object inventory")
for n in sorted(list(globals())):
    if re.search(r"(?i)(kernel|impulse|dyn.*(fit|coef|effect)|ds_rates|panel|target.*effect)", n):
        a = globals()[n]
        if isinstance(a, (np.ndarray, pd.DataFrame)):
            print(n, type(a).__name__, a.shape,
                  list(a.columns)[:14] if isinstance(a, pd.DataFrame) else "")
        elif isinstance(a, dict):
            print(n, "dict keys", [str(k) for k in list(a)[:12]])
        elif callable(a):
            print(n, "function args", a.__code__.co_varnames[:a.__code__.co_argcount])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 290\nForward31 corrected candidate failed: NLL 1.61612 versus 1.53191 (+.08421), concentrated in MAP. Rejected and replaced with gains [-3,.4] using the same corrected feedback-constrained projection, relative to original b_in. Requires new factual gate. Counts: 290 Python runs, 31 forwards.\n")
```
