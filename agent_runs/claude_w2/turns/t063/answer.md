The gap input and the time decay are what matter:
- With the gap fed log(seconds)/10, finger glucose comes out about 5.4 mg/dL low.
- With log(minutes)/10 it comes out about 4.9 mg/dL high.
- Removing the time decay of h turns the MAP bias from about 0 into +8.
- The `last_numeric` layout and the kq/rate rows change little (the bin rows 0 are zero).

So the correct setting probably lies between these. The z-prior context and the view may also use different gap inputs, and the decay may use a different time unit.

Next I grid over three things:
- a scale on dt for the decay (0, 0.5, 1, 2);
- the unit inside log(dt/unit)/10 for the view gap (1, 10, 60 s);
- the same unit for the z-prior context (1, 60 s).

I look for one combination that removes the bias for all four codes at once.

```python
def run6(f8, fs=1.0, gu=1.0, cu=1.0, ty=1):
    decf = torch.exp(-(dt_*fs)[:,None] * torch.exp(Wd['core.flow.a'])[None,:])
    hm = (HH[P_]*decf[:,None,:]).mean(1)
    b_ = OUT[P_] + LN_(hm, 'core.out_norm')
    ctx_ = b_ + M((torch.log(dt_/cu)/10)[:,None], 'gap')
    view_ = b_ + M((torch.log(dt_/gu)/10)[:,None], 'gap')
    zmu = M(ctx_, 'z_prior')[:, :32]
    q_ = LN_(view_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_, wc
out6 = []
for fs, gu, cu in itertools.product([0.0, 0.5, 1.0, 2.0], [1.0, 10.0, 60.0], [1.0, 60.0]):
    out6.append((rep(run6(fC, fs, gu, cu)[0]), fs, gu, cu))
out6.sort(key=lambda t: t[0][0]); 
for o_ in out6[:8]: print(o_)
```