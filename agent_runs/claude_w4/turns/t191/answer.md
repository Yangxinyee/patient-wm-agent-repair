`step_cv` returns only the new slots y for each mode. I'll compute the relative error of y for each mode. For the best one I'll rebuild out myself as `out_norm(mean over slots of y)` to confirm it reaches the accuracy in my notes (R2 about 0.998).

```python
for nm in [n for n in NAMES if 'out_norm' in n and n.startswith('core')]: print(nm, SHAPES[nm])
def out_from_y(y):
    m = y.mean(1)
    w = load('core.out_norm.weight'); b = load('core.out_norm.bias')
    return torch.nn.functional.layer_norm(m, (192,), w, b, 1e-5)
for mode in ['ageKV','lnM_ageKV','lnM','noage']:
    y_ = step_cv(rw, E0[rw], htf[rw], mode)
    o_ = out_from_y(y_)
    print(mode, 'y err %.4f' % float((y_-yt[rw]).norm()/yt[rw].norm()), 'out err %.4f' % float((o_-ot[rw]).norm()/ot[rw].norm()))
```