The per-reading breakdown shows how large v7's factual perturbations are at these gains:

| Readings | Prediction shift (rms) | Original residual sd |
|---|---|---|
| Finger glucose | 15.3 mg/dL | 15.0 |
| Lab glucose | 21.7 mg/dL | 12.8 |
| Arterial MAP | 6.6 mmHg | 5.0 |

So the edit moves predictions by about as much as their own error, while only a small, slowly building part of each pulse carries into the rollouts.

The shifts correlate positively with the original model's residuals (0.12–0.20). Part of the injected dose effect is genuinely in the data, beyond what the original model uses. The NLL cost therefore has a helpful first-order term, not just a quadratic penalty.

I fitted a rough model per reading type, with cost as a linear plus a quadratic term in a gain scale s, where s = 1 is the tested gains. It predicts:
- the cost is about zero near s ≈ 0.25–0.3;
- it stays within the 0.005 budget up to s ≈ 0.3–0.35.

At s = 0.3 the insulin x2 effect would be about −4.1 mg/dL against a truth of −13. That is only about 30% of the gap, but well within the fit constraint if the model holds.

Next I build v7 at s = 0.3 (insulin gain 1.455, NE gain 2.07). Then I measure factual NLL and the dose effects on sel100. This also gives a second point on the cost curve to fit it exactly.

**What would confirm it:** a total value-NLL increase near +0.002 (at most +0.004), with the dose effects scaling linearly from the gain-1 results. A cost near +0.023, the pure-quadratic scaling, would mean the helpful linear term is not there.

```python
kI,kN=1.455,2.07
kvn,Won,Wbn=build7(B,V5['tI5'],V5['tN5'],kI,kN)
for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
    save_edit(n,torch.tensor(x,dtype=torch.float32))
print('saved v7(%.3f,%.3f)'%(kI,kN), edited())
with open('/work/notes.md','a') as f:
    f.write('- fwd_12 = factual both sel100, v7(4.85,6.9): NLL 1.4938 -> 1.7479 (+.254). finger +.166 lab +.393 art +.374 cuff +.175.\n'
            '  dp rms finger 15.3 (res sd 15.0), lab 21.7 (12.8), art 6.6 (5.0), cuff 5.7 (7.7); corr(dp,res0) +.135/.148/.203/.116 => first-order term helps.\n'
            '  Rollout persistence of injected pulses is weak (~20%/h of immediate). Cost model per event dNLL(s)=a s+b s^2 (s=k/k_full) -> zero near s~.25-.3.\n'
            '- Run 285: SAVED v7(1.455,2.07) (s=.3). Next: factual + dose_edit both sel100.\n')
```