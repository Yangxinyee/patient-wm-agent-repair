This is the most informative result so far. Rate decoding, stage by stage (factual R²; arm-change slopes for x2, x1/2, x0, x3 from 6, x1.5 from 16, random multipliers):

| Stage | R² | Slopes |
|---|---|---|
| u (encoder output for the drug atoms) | 0.93 | 1.03, 0.96, 0.67, 0.95, 1.04, 0.84 |
| Prelude output, slot mean | 0.69 | about 1 for the x2, x1/2 and x1.5 arms |
| hn (768) | 0.57 | about 1 for the same arms |
| New state y, 4 slots (768) | 0.60 | about 1 for the same arms |
| out = LN(slot mean of y) (192) | 0.50 | 0.66, 0.88, 0.58, 0.48, 0.73, 0.47 |

- **The state does track counterfactual dose changes almost linearly.** The full 4-slot state shows it.
- **The step from y to `out` loses linearity.** That step is the slot average followed by LayerNorm, and the head only ever sees `out` and the decayed slot mean.
- **Precision is lost earlier.** It drops from u (R² 0.93) to the prelude (0.69), so factual dose deviations are not precisely decodable anywhere past u.
- **Rate 0 (the x0 arm) is out of the training range,** even for u.

Before designing a relay I need to know which step is responsible: the averaging over slots (signals cancelling across slots) or the final LayerNorm. I'll decode from:
- mean(y) before the LayerNorm;
- each slot separately;
- the decayed state the head sees at the next reading (dt ≈ 55 min), through the head's own normalisation.

I'll also check the shape of core.flow.a.
- If mean(y) keeps slope about 1, a small edit to the output normalisation path could restore a clean channel.
- If only some slots carry the signal, a relay in the coda would have to move it.

```python
print('flow.a shape', SHAPES['core.flow.a'], 'c_out', SHAPES['core.c_out.weight'])
Y=CS['y'].reshape(-1,4,192)
my=Y.mean(1)
print('mean(y) preLN', cv_eval(my, rate_arm))
for s_ in range(4): print('slot',s_, cv_eval(Y[:,s_], rate_arm))
fa=torch.exp(W['core.flow.a']).numpy()
dec=np.exp(-3300.0*fa); print('decay factor at 55min: min %.3g med %.3g max %.3g'%(dec.min(),np.median(dec),dec.max()))
hfm=(Y*dec.reshape((1,)+dec.shape) if dec.ndim>1 else Y*dec[None,None,:]).mean(1)
lnhf=LN(torch.as_tensor(hfm,dtype=torch.float32),W,'core.out_norm').numpy()
print('LN(mean hf) next', cv_eval(lnhf, rate_arm))
print('out+LN(mean hf)', cv_eval(CS['out']+lnhf, rate_arm))
print('[out,LNhf] concat', cv_eval(np.hstack([CS['out'],lnhf]), rate_arm))
# how much does the rate signal in mean(y) align with the mean/scale removed by LN?
from numpy.linalg import lstsq
Xc=my-my.mean(0); bcoef=lstsq(np.c_[Xc,np.ones(len(Xc))][:,:-1], rate_arm-rate_arm.mean(), rcond=None)[0]
print('std across dims of mean(y): mean %.3f sd %.3f'%(my.std(1).mean(), my.std(1).std()), 'corr(rate, row-std)', np.corrcoef(rate_arm, my.std(1))[0,1], 'corr(rate,row-mean)', np.corrcoef(rate_arm, my.mean(1))[0,1])
```