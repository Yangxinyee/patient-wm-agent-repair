v7 at gains 1 and 1 behaves as predicted. On top of the original model it adds:

| Edit | Added effect at gain 1 |
|---|---|
| Insulin x2 | −2.60 mg/dL |
| Insulin x1/2 | +1.20 mg/dL |
| NE x2 | +0.57 mmHg |
| NE x1/2 | −0.50 mmHg |

**Insulin.** The insulin response is close to linear in dose. A gain of about 4.85 would hit the truth on sel100: x2 at about −12.96 (truth −12.96) and x1/2 at about +6.0 (truth +6.48).

**NE.** The NE response is lopsided: halving moves MAP almost as much as doubling, as if the NE readout scales with log-rate. No single gain fits both:
- Gain 8.7 matches x2, but x1/2 then comes out at about −4.6 against a truth of −2.47.
- The least-squares compromise is about 6.9, giving x2 at about +3.9 and x1/2 at about −3.65.

**Factual fit.** By my rough estimate, full-strength injections may cost about 0.005 nats each for insulin and NE, so the factual fit may force smaller gains.

**This run:**
- Check the per-hour shape of the insulin and NE effects and whether rollouts still drift away from the original's.
- Build and save v7(4.85, 6.9).

After that I run factual and dose_edit passes on these weights. The per-event NLL changes will separate the insulin cost (finger/lab readings) from the NE cost (arterial/cuff readings), so I can see how far each gain must come down.

```python
de=pd.read_csv('/work/fwd_11/dose_edit_edited.csv'); do=pd.read_csv('/work/fwd_11/dose_edit_original.csv')
m=de.merge(do,on=['stay_id','hour','edit'],suffixes=('_e','_o'))
print('edits:',list(m.edit.unique()))
for ed in ['insulin_x2','ne_x2']:
    x=m[(m.edit==ed)&(m.reached_e.astype(bool))]
    g=x.groupby('hour')
    print(ed,'added shift by hour',(g['shift_e'].mean()-g['shift_o'].mean()).round(2).tolist())
    print(ed,'drift factual e-o by hour',(g['factual_e'].mean()-g['factual_o'].mean()).round(2).tolist())
kI,kN=4.85,6.9
kvn,Won,Wbn=build7(B,V5['tI5'],V5['tN5'],kI,kN)
for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
    save_edit(n,torch.tensor(x,dtype=torch.float32))
print('saved v7(%.2f,%.2f)'%(kI,kN))
with open('/work/notes.md','a') as f:
    f.write('- fwd_11 = dose_edit both sel100, v7(1,1): EDITED ins x2 -2.948, x1/2 +1.411, NE x2 +.584, x1/2 -.742 (orig -.348/+.208/+.018/-.245).\n'
            '  => added per k: ins -2.60/+1.20 (linear), NE +.566/-.497 (log-like asym). kI~4.85 fits ins; NE compromise kN~6.9 (x2 3.9, x1/2 -3.65).\n'
            '  Rough factual-cost estimate at full gains ~.005 (ins) + ~.0045 (NE) -> may need to scale down. Constant centering needed for sustained effect.\n'
            '- Run 283: SAVED v7(4.85,6.9). Next: factual both sel100 + dose_edit both sel100.\n')
```