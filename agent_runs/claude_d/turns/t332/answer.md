Run 293 set up the cost model and saved E8. The glucose-cost model, calibrated on sel1, predicts the other three sets within about 0.0003: R1 0.0008 vs 0.0011 measured, R2 0.0016 vs 0.0013, R3 0.0002 vs 0.0005. That is good enough to trust nearby projections.

**Choosing the gains.** The automatic choice saved as E8 is insulin 0.95 and NE 0.5: projected pooled +0.0010, worst set (sel1) +0.0040. Insulin 1.0 with NE 0.4 projects almost the same cost (pooled +0.0009, worst set +0.0039). It brings the main readout, insulin x2, to my estimate on sel1: −13.9 against −14.0, compared with −13.3 for E8. It also gives x1/2 of 7.1 against 7.0. What it gives up is small NE gains (x2 0.50 against 0.64, x1/2 −1.15 against −1.32). My NE targets are also much less certain than my insulin ones, especially the part of the x2 effect beyond the recorded rate cap. So I prefer full insulin gain with NE at 0.4.

**Risk.** Sets of 100 stays vary by about 0.0017. For an evaluation set of 100 stays or more, the chance of exceeding 0.005 is roughly 2% or less.

Next I build E9 (insulin 1.0, NE 0.4, encoder kept), save it, and log the decision. I will then measure E9 on sel1, the worst set.
- About +0.004 or below confirms the projection.
- Clearly above +0.0045 would mean scaling NE back to 0.3 or dropping it.

```python
import numpy as np, torch
z3=np.load('/work/E3b.npz'); z4=np.load('/work/E4.npz'); z7=np.load('/work/E7h.npz')
W1,b1,W2,b2=[z7[f'arr_{i}'].astype(np.float32).copy() for i in range(4)]
bf=lambda x: torch.tensor(x).to(torch.bfloat16).float().numpy()
fi,fn=1.0,0.4
W2[:,[31,143]]=bf(z3['arr_2'][:,[31,143]]*fi); W2[:,[124,142]]=bf(z4['arr_2'][:,[124,142]]*fn)
hn=['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']
for n,a in zip(hn,[W1,b1,W2,b2]): save_edit(n, torch.from_numpy(np.ascontiguousarray(a)))
np.savez('/work/E9h.npz',W1,b1,W2,b2)
O2=np.asarray(load_original(hn[2])); chg=np.where(np.abs(np.asarray(load(hn[2]))-O2).max(0)>0)[0]
print('saved E9; W2 cols changed', chg, 'ins col max %.3f ne col max %.3f'%(np.abs(W2[:,31]).max(), np.abs(W2[:,124]).max()), 'n edited', len(edited()))
with open('/work/notes.md','a') as fh: fh.write("- E8 (ins .95, NE .5) proj pooled .0010 max .0040 -> replaced by E9 = ins f=1.0, NE f=0.4 (+E5 enc), proj pooled .0009, max set sel1 .0039; proj readouts sel1 ins -13.9/+7.1, NE x2 ~.5, half ~-1.15. /work/E9h.npz = current. Next: factual sel1, dose_edit sel1/sel2.\n")
```