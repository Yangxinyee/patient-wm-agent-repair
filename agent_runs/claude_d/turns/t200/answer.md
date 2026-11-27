E2 corrects the factual deficits as intended. After the edit the insulin residual slopes are 0.18, 0.39, -0.49 (from -1.6, -1.33, -0.39). The NE lag-1 slope falls from 3.15 to 0.62.

But it costs +0.0063 NLL overall, over the 0.005 limit:

| code | ΔNLL |
|---|---|
| finger glucose | +0.012 |
| lab glucose | +0.005 |
| arterial MAP | +0.005 |
| cuff MAP | +0.003 |

The insulin injection has sd 2.9 mg/dL on present rows, against about 1.3 of true signal. Most of the excess comes from the noisy lag-2 decoder and from the uneven gain of the exponential tilt. There is also some leakage (sd 0.86) on rows without recent insulin.

Before tuning the trade-off I need to know how the rollout actually responds to these injections. Only the tool can tell me that: whether feedback through sampled readings builds the cumulative effect, and how the decoders respond to doubled or halved doses. So I am saving E2 as the current edit (an informative over-budget version) and will run the dose_edit comparison next.

```python
import torch
W1e,b1e,W2e,b2e = E2
for nm, arr in [('attr_heads.value_bin.0.weight',W1e),('attr_heads.value_bin.0.bias',b1e),('attr_heads.value_bin.2.weight',W2e),('attr_heads.value_bin.2.bias',b2e)]:
    save_edit(nm, torch.tensor(arr.astype(np.float32)))
chk = [float(np.abs(load(n).numpy().astype(np.float64)-a).max()) for n,a in [('attr_heads.value_bin.0.weight',W1e),('attr_heads.value_bin.2.weight',W2e)]]
print(edited(), chk)
sel1 = json.load(open('/work/sel1.json')); print(len(sel1), type(sel1[0]).__name__)
with open('/work/notes.md','a') as f:
    f.write("""
- Runs 166-179: gated-channel head edit. Repurpose hidden units (31,143)=insulin, (54,9)=NE (joint ablation dNLL +.0003, 4-draw).
  unit a: M*(AND gate - th) + s/lam, unit b: M*(AND gate - th); W2 cols +-lam*t (exp tilt, cen/cov). AND gate = family gate + presence(drug at t-1) gate (ridge on z-aug wc).
  Decoders on present rows (z-aug ridge): ins eps1 R2 .59 k .57, eps2 .24; NE D1 .19 k .25. Present-row deficits: ins -1.6,-1.33,-0.39; NE joint 3.15,.53,-2.13 (D1 only 2.17).
  E2 (a_ins=(-2.8,-5.5), a_ne=8.7): offline dNLL +.0063 (gf +.0117) -> over budget; resid slopes fixed. Saved as current edit for dose_edit test (fwd 8).
""")
```