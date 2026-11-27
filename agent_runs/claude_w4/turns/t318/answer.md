Forward pass 10 shows that removing the leak fixed v6's sign and linearity, but the effect is far too small:

| Edit (sel100) | Original | v6(1,1) | Truth |
|---|---|---|---|
| Insulin x2 → finger glucose | −0.35 | −1.42 | −12.96 |
| Insulin x1/2 → finger glucose | +0.21 | +0.70 | +6.48 |
| NE x2 → arterial MAP | +0.02 | +0.10 | +4.94 |
| NE x1/2 → arterial MAP | −0.25 | −0.37 | −2.47 |

The edit adds about −1.07 for insulin x2 and +0.49 for x1/2, close to the −½ ratio a linear effect should give. Its NE effect is only about +0.08 / −0.13.

At the design's own scaling, the insulin channel delivers only about 18% of the intended rollout effect, and the NE channel about 3%. Matching the truth would take gains of about 12 for insulin and about 58 for NE. Those are large pushes on the state, with risk of cross-talk and nonlinearity.

**Likely cause.** The targets are reading-surprise directions in the new slots, and the prediction head reads them only weakly. The v5 parts built in the unrecorded runs look like the fix: targets set through linear readouts of the predicted means (`wF`, `wL`, `wA`, `wC`), using the measurement-bundle normalisation scale (`sig_m`).

**This run:**
1. Briefly check fwd_10's per-hour shape and whether rollouts drift.
2. Compare how strongly the v4 and v5 targets move each predicted mean, including cross-talk such as NE moving glucose.
3. Build and save v7: the leak-free readouts with the v5 b_in targets, gains 1 and 1.

**What would support v5:** its targets move their own reading about 5–30 times more than v4's per unit, with little cross-talk.

```python
import pandas as pd
de=pd.read_csv('/work/fwd_10/dose_edit_edited.csv'); do=pd.read_csv('/work/fwd_10/dose_edit_original.csv')
print('cols',list(de.columns),len(de))
for col in de.columns:
    if de[col].dtype==object: print(col,'vals',list(de[col].unique())[:8])
V5=np.load('/work/v5_parts.npz')
W={k:V5[k] for k in ['wF','wL','wA','wC']}
for nm,t in [('tgt4_I',V4['tgt_I']),('tgt5_I',V5['tgt5_I']),('tgt4_N',V4['tgt_N']),('tgt5_N',V5['tgt5_N'])]:
    print(nm,'|t| %.3f'%np.linalg.norm(t),' '.join(f'{k}:{float(W[k]@t):.3f}' for k in W))
print('|w|',{k:round(float(np.linalg.norm(v)),4) for k,v in W.items()},'sig_m',float(V5['sig_m']))
def build7(B,tIv,tNv,kI,kN,rbI=float(V4['rbar'])):
    kv=f64(kv0).copy(); kv[240:288]+=A_@M_@B.T
    Wo=f64(Wo0).copy(); Q3=np.stack([V4['qI'],V4['qN'],-rbI*V4['qI']],1)
    Wo[:,48:96]=f64(Wo0[:,48:96])@(np.eye(48)-A_@A_.T)+c_*Q3@Mi_@A_.T
    Wb=Wb0.copy()
    for k,q,t in [(kI,V4['qI'],tIv),(kN,V4['qN'],tNv)]: Wb+=np.outer(k*t-Wb0@(pnw*q),q/pnw)
    return kv,Wo,Wb
kvn,Won,Wbn=build7(B,V5['tI5'],V5['tN5'],1,1)
for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
    save_edit(n,torch.tensor(x,dtype=torch.float32))
print('saved v7(1,1)')
with open('/work/notes.md','a') as f:
    f.write('- fwd_10 = dose_edit both sel100, v6(1,1): EDITED ins x2 -1.422, x1/2 +.696, NE x2 +.102, x1/2 -.374; ORIG -.348/+.208/+.018/-.245.\n'
            '  => leak fixed, sign right & ~linear, but chain gain only ~.18 (ins) / ~.03 (NE) of design. Need kI~12, kN~58 with v4 targets.\n'
            '- Run 282: v7 = build7(B, tI5, tN5, kI, kN): v5 targets (predicted-mean readouts wF..wC, sig_m 58.6). SAVED v7(1,1). Next: dose_edit sel100.\n')
```