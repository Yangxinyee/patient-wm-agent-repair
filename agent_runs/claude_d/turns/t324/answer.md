Run 290 settles the decision.

- **The quadratic fit is not usable as an absolute cost.** It predicts +0.0020 at full gain, but the measured glucose cost is +0.0040. Calibrated to the measured value, with the 0.22 slope, the overall cost of the insulin channel scales roughly as 0.0069·f² − 0.0030·f. That gives about +0.002 at f = 0.8 and +0.004 at f = 1.
- **fwd13 (E5, sel1)** confirms the encoder fix behaves: NE x2 rises 1.03 → 1.35, insulin is unchanged, and fwd15 and fwd16 agree to 0.001 in NLL. So the encoder costs essentially nothing factually.

New edit, E6:
- **NE head channel removed** (back to E3b, which also restores the shared output bias). Its map_arterial shifts are pure noise on tuning stays.
- **E5 encoder units kept.** They are free factually and may lift NE x2 slightly.
- **Insulin channel output scaled to f = 0.8.** I scale its two output columns, rounded to bf16 so they remain exact opposites.

I will first check that E3b changed only the two insulin units. After that I will save the edit, log the finding in the notebook, and print the sel1 stay ids for the factual check.

How to read the factual check that follows:
- About +0.002 on sel1 confirms the scaling model.
- Above +0.003 means I need to scale down further.

```python
import numpy as np, torch, json
f=0.8
z3=np.load('/work/E3b.npz'); z5=np.load('/work/E5enc.npz')
hn=['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']
en=['encode_action.mlp1.0.weight','encode_action.mlp1.0.bias','encode_action.mlp1.2.weight','encode_action.mlp1.2.bias']
O={n:np.asarray(load_original(n)) for n in hn}
W1,b1,W2,b2=[z3[f'arr_{i}'].astype(np.float32).copy() for i in range(4)]
rw=np.where(np.abs(W1-O[hn[0]]).max(1)>0)[0]; cols=np.where(np.abs(W2-O[hn[2]]).max(0)>0)[0]
print('W1 rows', rw, 'b1', np.where(np.abs(b1-O[hn[1]])>0)[0], 'W2 cols', cols, 'b2', float(np.abs(b2-O[hn[3]]).max()))
if len(cols)==2: print('antisym', float(np.abs(W2[:,cols[0]]+W2[:,cols[1]]).max()), 'colnorm', float(np.abs(W2[:,cols[0]]).max()))
W2n=W2.copy(); W2n[:,cols]=torch.tensor(W2[:,cols]*f).to(torch.bfloat16).float().numpy()
def sv(n,a): save_edit(n, torch.from_numpy(np.ascontiguousarray(np.asarray(a,dtype=np.float32))))
for n,a in zip(hn,[W1,b1,W2n,b2]): sv(n,a)
for i,n in enumerate(en): sv(n,z5[f'arr_{i}'])
np.savez('/work/E6h.npz', W1, b1, W2n, b2)
print('edited', [e[-22:] for e in edited()], 'chk', float(np.abs(np.asarray(load(hn[2]))-W2n).max()), float(np.abs(np.asarray(load(en[0]))-z5['arr_0']).max()))
with open('/work/notes.md','a') as fh:
    fh.write("\n## Restart 11 (runs 287-)\n- fwd15/16 = factual both sel1 (E5 and E4): edited 1.52200 vs 1.51191 = +.0101 OVER BUDGET. gf +.0117 (shift~resid slope .22), map_a +.0193 (slope .03 = pure noise), gl +.0013, map_c -.0015. Offline cost estimates were ~4x too optimistic (decoders do not generalize to tuning stays).\n- Encoder (E5) factual effect negligible (15 vs 16 identical). fwd13 = E5 dose_edit sel1: ins -13.91/+7.08, ne_x2 1.35, ne_half -2.15.\n- DECISION: drop NE head channel; insulin channel W2 cols scaled f; keep E5 encoder. E6 = E3b(W2 ins cols*0.8, bf16) + E5enc, saved (/work/E6h.npz). Expected factual ~+.002 (cost(f)~.0069f^2-.0030f overall), ins readouts ~ orig + f*(E4-orig).\n")
s1=json.load(open('/work/sel1.json')); print(type(s1).__name__, len(s1)); print(json.dumps(s1 if isinstance(s1,list) else s1)[:1200])
```