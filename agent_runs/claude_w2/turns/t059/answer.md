The value head now matches the tool's predicted means to about 5.5 mg/dL for glucose and about 2 mmHg for MAP. That is good, but not yet exact.

Two leads:
- Putting has_last in columns 0, 3 and 6 of `last_numeric` improved the fit, so the 8-column layout is still not fully pinned down.
- `last_status` row 1 is a large vector (norm 13.2) while row 0 is zero. It is probably switched on whenever a previous reading exists.

Next: test status = has_last, then run a greedy column-by-column search over the `last_numeric` inputs (has, log-age, last z, constant one, zero). I start from the current best layout and keep any change that lowers the mean absolute error.

If the remaining gap is mostly due to using the z prior mean instead of 4 sampled z's, the error will stall at about 2–3 mg/dL. That is acceptable for calibrating the readout.

```python
def run3(f8, status_vec=None, ty=1):
    ctx_ = base0 + gapv('log10')
    zmu = M(ctx_, 'z_prior')[:, :32]
    q_ = LN_(ctx_ + L(zmu, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
    wc = q_ + Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t] + Wd['embed.bins.rate.weight'][0]
    if status_vec is not None: wc = wc + Wd['last_status.weight'][torch.tensor(status_vec)]
    lg = M(wc, 'attr_heads.value_bin') + Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    p_ = torch.softmax(lg, -1).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_, wc
la_h = np.log1p(fa4.age_s.values/3600.0)/10 * fa4.has.values
feats = {'zero': np.zeros(len(fa4)), 'has': fa4.has.values, 'la': la_h, 'zl': fa4.zl.values, 'one': np.ones(len(fa4)),
         'nohas': 1-fa4.has.values, 'la_s': np.log1p(fa4.age_s.values)/10*fa4.has.values}
assign = ['has','la','zl','has','zero','zero','has','zero']
stat_opts = {'none': None, 'has': fa4.has.values.astype(int), 'nohas': (1-fa4.has.values).astype(int)}
def build(asg): return np.stack([feats[a_] for a_ in asg], 1)
best_s = None
for sk, sv in stat_opts.items():
    e_ = sc(run3(build(assign), sv)[0]); print('status', sk, e_)
    if best_s is None or e_ < best_s[0]: best_s = (e_, sk)
sv_best = stat_opts[best_s[1]]
for it_ in range(2):
    for c_ in range(8):
        cand = []
        for fk in feats:
            a2 = assign.copy(); a2[c_] = fk; cand.append((sc(run3(build(a2), sv_best)[0]), fk))
        cand.sort(key=lambda t: t[0]); assign[c_] = cand[0][1]
    print(it_, assign, cand[0][0])
```