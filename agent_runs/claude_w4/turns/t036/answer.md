## Step 33 result: the model's dose effects are 15–40 times too small

**What the last result showed.**
- The summary readout is the pooled mean over reached stay-hours.
- "Reached" means the hour comes after the first dose at or after hour 12 (this matches exactly).
- Applying my record-based kernels to the recorded doses of these stays gives the true targets:

| edit | model now | my true estimate |
|---|---|---|
| insulin ×2 (finger glucose) | −0.35 | −13.0 mg/dL |
| insulin ×1/2 | +0.21 | +6.5 |
| norepinephrine ×2 (arterial MAP) | +0.02 | +4.9 mmHg |
| norepinephrine ×1/2 | −0.25 | −2.5 |

**What this means.** The model is close to dose-blind for both drugs. The edit has to add a large, correctly timed causal pathway.

**Choosing how to edit.** The dose reaches the predictions only through a deep nonlinear path (action encoder → prelude attention → recurrent block → coda). Hand-designed edits calibrated only with forward passes would be crude. A much stronger route is to re-implement the model locally in torch, check it against the tool (states and factual means), and then design and test edits offline.

**What I will do next.** Read tensors.json for implementation details: activations, norms, which categorical index each atom uses.

**What would count as evidence.** If the metadata documents the layer types, a faithful local re-implementation is feasible. If not, I will fall back to a regression-based approach on states.

```python
tn = json.load(open('/model/tensors.json'))
ks = list(tn.keys())[:5]
for k in ks: print(k, tn[k])
# any non-tensor metadata?
print([k for k in tn if k not in NAMES][:20])
import re
print(open('/model/ARCHITECTURE.md').read().count('\n'))
# check for other files in /model
print(os.listdir('/model'))
```