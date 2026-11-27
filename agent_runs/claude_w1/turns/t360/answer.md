**What the last result showed:** Raising the numeric dose/rate scaling from 3 to 10 gained nothing:

| Readout | Factor 3 | Factor 10 |
|---|---|---|
| Insulin x2 | −2.09 | −1.94 |
| Insulin x1/2 | +1.96 | +1.88 |
| Norepinephrine x2 | +0.18 | +0.20 |
| Norepinephrine x1/2 | −0.39 | −0.32 |

**What I conclude:** This input pathway saturates. My notes record a LayerNorm right after the dose encoding, which would explain the saturation. Factor 3 is slightly closer to my record-based estimates on three of the four readouts. It is also the only version whose factual fit I have checked, and that check showed no cost. So I will go back to factor 3, rebuilt from the original weights, and log this in the notebook.

**Earlier levers I have ruled out:**
- The head-level readout corrections failed on the tool.
- Amplifying the model's internal dose direction was rejected earlier: LayerNorm after the prelude, and an inconsistent sign.

**What would confirm the revert:** columns 2 and 4 at ratio 3 and everything else unchanged. The weights would then match the version that passed both earlier checks (passes 11 and 12).

```python
import numpy as np, torch
W0=np.asarray(load_original('embed.numeric.weight')).copy()
W=W0.copy(); W[:,[2,4]]*=3.0
save_edit('embed.numeric.weight', torch.from_numpy(W.astype(np.float32)))
W2=np.asarray(load('embed.numeric.weight'))
print('ratio',(np.linalg.norm(W2,axis=0)/np.linalg.norm(W0,axis=0)).round(3), edited(), float(np.abs(W2-W).max()))
with open('/work/notes.md','a') as f:
    f.write("- fwd_13 (dose_edit both setA, x10): ins x2 -1.94, half +1.88, ne x2 +.20, half -.32 -> SATURATED (no gain over x3).\n"
            "- Run 294: reverted to x3 on cols [2,4] (verified: factual neutral fwd_12, readouts fwd_11). FINAL candidate.\n")
```