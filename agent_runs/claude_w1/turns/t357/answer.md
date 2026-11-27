**What the last result showed:** Tripling the numeric dose and rate columns left the factual fit unchanged: NLL 1.51901 against 1.51903 for the original, with every reading type within ±0.001. That makes sense. These two columns only feed the drug-start atoms, and the model barely uses the continuous dose value when it predicts readings.

**What I conclude:** This lever is weak, but the factual fit gives me plenty of room. So I can scale these columns much harder and look for a larger insulin effect. Norepinephrine did not respond at a factor of 3. It may stay unresponsive, but scaling should not hurt it.

**Next step:** Set the factor to 10, applied to the original weights. In this run I also check which columns the current edit scaled, and I log the results in the notebook. Then I will run a dose-edit comparison and a factual comparison on setA.

**What would confirm or refute:**
- *Confirm:* the insulin readouts move clearly further toward the record-based targets (about −14 for x2, +7 for x1/2) and the factual NLL stays within +0.005.
- *Refute:* the readouts saturate or turn erratic, or the factual fit degrades. Then I fall back to the factor-3 edit.

```python
import numpy as np, torch
W0=np.asarray(load_original('embed.numeric.weight')).copy()
Wc=np.asarray(load('embed.numeric.weight'))
r=np.linalg.norm(Wc,axis=0)/np.linalg.norm(W0,axis=0)
print('current ratio',r.round(3))
cols=[i for i in range(6) if abs(r[i]-1)>1e-3]
print('cols',cols)
s=10.0
W=W0.copy(); W[:,cols]*=s
try:
    save_edit('embed.numeric.weight', torch.from_numpy(W.astype(np.float32)))
except Exception as e:
    print('torch save failed',e); save_edit('embed.numeric.weight', W.astype(np.float32))
W2=np.asarray(load('embed.numeric.weight'))
print('new ratio',(np.linalg.norm(W2,axis=0)/np.linalg.norm(W0,axis=0)).round(3), edited())
with open('/work/notes.md','a') as f:
    f.write("- fwd_11 (dose_edit both setA, numeric cols x3): ins x2 -2.09 (orig -1.45), ins half +1.96 (+.82), ne x2 +.18 (+.15), ne half -.39 (-.41). Weak lever, NE unmoved.\n"
            "- fwd_12 (factual both setA, x3): NLL 1.51901 vs 1.51903 -> no factual cost (numeric dose cols only feed drug START atoms).\n"
            f"- Run 293: numeric cols {cols} scaled x{s} (from original). Next: dose_edit both + factual both on setA.\n")
```