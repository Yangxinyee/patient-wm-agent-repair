The structure of my error points to the register terms:
- My finger-glucose predictions sit about 5 mg/dL too low across the board.
- My MAP predictions follow the last reading too weakly: the error correlates −0.78 with the last value's z-score.

So the issue is in how the last value or a constant term enters the head input, not in z.

Candidate explanations, all tested in this run:
- The `bins.rate[0]` row I add may not belong in the input.
- An `embed.kq` row may be added.
- The gap input may use another unit.
- The view may use h before the time decay.
- The z-score of the last value may sit in a different column of `last_numeric`.

Whichever variant removes both the glucose bias and the MAP persistence error is the right one.

```python
print('row0 norms: rate', float(Wd['embed.bins.rate.weight'][0].norm()), 'value', float(Wd['embed.bins.value.weight'][0].norm()), 'dose', float(Wd['embed.bins.dose.weight'][0].norm()), 'kq', Wd['embed.kq.weight'].norm(dim=1).numpy().round(2))
def run5(f8, gfun=lambda d: torch.log(d)/10, use_rate0=True, kq=None, noflow=False, ty=1, extra=None):
    hm = (HH[P_] if noflow else hfl).mean(1)
    ctx_ = OUT[P_] + LN_(hm, 'core.out_norm') + M(gfun(dt_)[:,None], 'gap')
    zmu = M(ctx_, 'z_prior')[:, :32]
    q_ = LN_(ctx_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    if use_rate0: wc = wc + Wd['embed.bins.rate.weight'][0]
    if kq is not None: wc = wc + Wd['embed.kq.weight'][kq]
    if extra is not None: wc = wc + extra
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_
fC = build(['has','la','zl','has','zero','zero','zero','zero'])
def rep(pm_): return (sc(pm_),) + tuple(fa4.assign(ad=pm_-tgt).groupby('event').ad.mean().round(2).values)
print('base', rep(run5(fC)[0]))
print('no rate0', rep(run5(fC, use_rate0=False)[0]))
for kq in range(4): print('kq', kq, rep(run5(fC, kq=kq)[0]))
print('noflow', rep(run5(fC, noflow=True)[0]))
for nmg, gf_ in [('log', torch.log), ('log_min/10', lambda d: torch.log(d/60)/10), ('log1p/10', lambda d: torch.log1p(d)/10), ('log_h', lambda d: torch.log(d/3600)), ('log_min', lambda d: torch.log(d/60))]:
    print(nmg, rep(run5(fC, gfun=gf_)[0]))
for c_ in [4,5,6,7]:
    a2 = ['has','la','zl','has','zero','zero','zero','zero']; a2[c_]='zl'; print('zl also col', c_, rep(run5(build(a2))[0]))
```