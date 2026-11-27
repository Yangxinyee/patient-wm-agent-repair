On these 100 stays the residual-on-deviation coefficients are noisy (se ≈ 0.5). Their slope is 0.42, and the correction at half strength improves glucose NLL (−0.0015) while full strength worsens it (+0.0009). The 2000-stay estimates (se 0.14) are the reliable ones, and they say the model misses the dose effects in factual prediction.

The better formulation is r = A(exact dose history) + B(head input). A comes from a clean dose channel with the rollout filter G′. B is a ridge fit of the remaining factual residual on the head input, restricted so it does not react to dose changes. This avoids the noisy policy compensation altogether. The hard part is A: an isolated coordinate of the recurrent state that carries an exponentially decaying dose sum.
- The dose enters through the prelude (u token → slot → prelude_norm → b_in).
- It must not be rewritten by the blocks, c_out or coda.
- It must be read only by a new value-head unit.

To design those edits I need the exact list of tensors and shapes for the core, prelude, blocks, coda and heads. Next I print them compactly, plus the per-dimension decay parameters (flow.a, a_in, delta_in).

```python
import collections
grp=collections.OrderedDict()
for nme in NAMES:
    p=nme.split('.')[0] if not nme.startswith('core') else '.'.join(nme.split('.')[:2])
    grp.setdefault(p,[]).append(nme)
for p,l in grp.items():
    print(p,':',', '.join(f"{x[len(p)+1:] if x!=p else x}{tuple(SHAPES[x])}" for x in l))
fa=torch.exp(W['core.flow.a']); sp_=F.softplus(W['core.delta_in']); ab=torch.exp(-sp_*torch.exp(W['core.a_in']))
print('flow.a exp: min %.2e med %.2e max %.2e'%(fa.min(),fa.median(),fa.max()), ' per-hour flow decay med',float(torch.exp(-3600*fa).median()))
print('sp: min %.3f med %.3f max %.3f'%(sp_.min(),sp_.median(),sp_.max()),' abar: min %.3f med %.3f max %.3f'%(ab.min(),ab.median(),ab.max()))
```