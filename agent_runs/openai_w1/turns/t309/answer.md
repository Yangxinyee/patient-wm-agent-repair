The feedback penalty reduced the measured loop substantially while preserving local glucose and MAP sensitivity. However, the reconstructed inputs do not exactly match the cached fitting inputs, so this is only provisional evidence. I’ll install a moderate-strength candidate and test factual fit on the same stays as the failed injection. A loss increase above 0.005 would reject this strength.

```python
import pandas as pd
from pathlib import Path

B271 = np.asarray(fitsB270[10000.0], dtype=np.float64)
P271 = np.asarray(fits266[1.0], dtype=np.float64)
gains271 = np.array([-1.0, 2.0], dtype=np.float64)
base_b271 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
delta_b271 = (B271*gains271)@P271.T
save_edit("core.b_in.weight", torch.as_tensor(base_b271+delta_b271, dtype=torch.float32))
candidate271 = {n: np.asarray(load(n)).copy() for n in edited()}

f271 = pd.read_csv("/work/fwd_28/factual_original.csv")
ids271 = [int(x) for x in f271.stay_id.unique()]
print("factual columns",list(f271.columns))
print("STAYS FOR PAIRED FACTUAL TEST:",ids271)
print("injection gains",gains271,"weight delta RMS",
      float(np.sqrt(np.mean(delta_b271**2))))
leak271 = np.asarray(factualE258)[hold263]@P271
print("held factual projection RMS",np.sqrt(np.mean(leak271**2,axis=(0,1))))
print("held instantaneous reading-shift RMS",
      np.sqrt(np.mean((leak271*gains271)**2,axis=(0,1))))
print("edited tensors",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 271\nInstalled candidate271: original core.b_in plus feedback-penalized B270[10000] @ diag([-1,2]) @ immediate-null P266[1].T; retained safe236 edits. Local feedback was substantially reduced with unit channel sensitivity retained, but reconstructed E mismatch means GPU factual validation is essential. Counts: 271 Python runs, 28 forwards.\n")
```
