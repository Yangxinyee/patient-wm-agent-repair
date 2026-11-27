The architecture doc fixes the pathway a written dose signal has to travel. The z posterior mean enters the state only as a prelude token (z_in), mixes into the slots (E), and then passes through the recurrent block and the coda. Heads read view → q → with_code, and the value head is the only place where glucose and MAP readings are produced.

Two points matter for the edit:
1. **Dose edits keep the drug atoms and only scale rate and dose.** So the x0 edit appears as rate-0 atoms, not as missing atoms. My writer must therefore treat an insulin atom with rate 0 as a strong negative shock relative to the expected dose.
2. **The correct correction is in terms of dose innovations.** The reasoning: the model is roughly calibrated on readings. It "explains" the dose effect through the readings, which predict the policy dose, and is nearly insensitive to the dose itself. The missing term is kappa_l × (dose − policy-expected dose given the information at dosing time).
   - That term is mean-zero in factual data, so the factual fit should improve, not degrade.
   - Under an edit, its dependence on the readings cancels the model's implicit "doses follow the policy" assumption.
   - kappa can then be obtained by regressing the model's factual errors on the written channel.

So the writer must compute the innovation itself:
- insulin: dose − expected dose given recent glucose;
- norepinephrine: rate − expected rate given the previous rate and MAP.

At an insulin or norepinephrine bundle the readings are only available through ctx (the state after the hour-h reading bundle). The key feasibility question is therefore how well ctx linearly predicts the dose: R² of rate on ctx compared with the policy's own R². Before testing that, this run inspects the structure of the saved state files and metadata (S5 and D5), so the test is built on the right rows.

```python
print('S5 files:', [(k, S5[k].shape, S5[k].dtype) for k in S5.files])
print('S2 files:', [(k, S2[k].shape) for k in S2.files])
print('D5 keys:', {k:(type(v).__name__, getattr(v,'shape',len(v) if hasattr(v,'__len__') else v)) for k,v in D5.items()})
k0 = S5.files[0]
for k in S5.files:
    a = S5[k]
    if a.ndim==1: print(k, a[:12])
print('glast', glast.shape, glast[0,:10]); print('gbest', gbest.shape, gbest[0,:10]); print('gf', gf[0,:10])
print('ins', ins[0,:10]); print('ne', ne[0,:10]); print('ids', list(ids)[:3] if hasattr(ids,'__iter__') else ids)
print(open('/data/README.md').read()[:3000])
```