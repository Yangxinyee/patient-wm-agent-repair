Run 105 shows the per-reading file layout: stay, hour, event, observed value, predicted mean and bin NLL, with 5673 rows.

Now I'll implement the value head on the recorded states:
- The view is ctx plus z_head of a prior sample of z.
- The query is query_norm(view + type_emb[type]).
- The register terms come from the last earlier reading of the same code: has_last, log(1+age)/10, its z-score and its bin.
- The value MLP output is added to the transition-table row, then a softmax over 17 classes. Probabilities are averaged over prior samples, and the predicted mean is the sum of p(b) times the bin centres.

First I'll scan the type index (0–14) and the last_status index on a subset of glucose_finger and map_arterial readings. I'll use many prior samples, so that the remaining mismatch reflects only the factual pass's own 4-sample noise. The right type index should give near-identical predicted means; I expect it to equal the code id.

```python
print('value_bin_centres type:', type(VOCAB['value_bin_centres']), (list(VOCAB['value_bin_centres'].keys()) if isinstance(VOCAB['value_bin_centres'], dict) else len(VOCAB['value_bin_centres'])))
vbc_ = VOCAB['value_bin_centres']
def centres_of(code):
    if isinstance(vbc_, dict): return np.array(vbc_[code], dtype=np.float64)
    return np.array(vbc_[code_id[code]], dtype=np.float64)
print('centres glucose_finger', centres_of('glucose_finger'))
def head_inputs(s_):
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]; times = np.array([b_[0] for b_ in bl_])*3600.0
    rows = []; last = {}
    for k1 in range(len(bl_)):
        if k1 >= 1:
            for a in bl_[k1][1]:
                if a['kind'] == 0 and a.get('value') is not None:
                    lv = last.get(a['code'])
                    rows.append((k1, a['code'], a['value'], lv, times[k1]-times[k1-1], (times[k1]-lv[1]) if lv else None))
        for a in bl_[k1][1]:
            if a['kind'] == 0 and a.get('value') is not None: last[a['code']] = (a['value'], times[k1])
    return rows, r0_, times
def head_probs(s_, type_idx, status_idx, ln_layout, nsamp=64, seed=0, codes=('glucose_finger','map_arterial')):
    rows, r0_, times = head_inputs(s_); g_ = torch.Generator().manual_seed(seed); res = []
    for (k1, c, val, lv, dt, age) in rows:
        if c not in codes: continue
        prev = Xt[r0_+k1-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
        base = out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
        pr = mlp_(base, 'z_prior', 'gelu'); zm, zls = pr[:32], pr[32:]
        zs = zm[None] + torch.exp(zls)[None]*torch.randn(nsamp, 32, generator=g_)
        view = base[None] + lin_(zs, 'z_head')
        q = ln_(view + T_['type_emb.weight'][type_idx], 'query_norm')
        lnin = torch.zeros(8)
        if lv is not None: ln_layout(lnin, c, lv[0], age)
        wc = q + T_['embed.code.weight'][code_id[c]] + lin_(lnin, 'last_numeric')
        lb = fbin(c, 'value', lv[0]) if lv is not None else 0
        wc = wc + T_['embed.bins.value.weight'][lb] + (T_['last_status.weight'][status_idx] if (lv is not None and status_idx) else 0)
        logits = mlp_(wc, 'attr_heads.value_bin', 'gelu') + T_['embed_transition.value_bin.weight'][code_id[c]*17 + lb]
        p = torch.softmax(logits, -1).mean(0).numpy()
        cen = centres_of(c); pm = float((p[1:]*cen).sum()); nll = float(-np.log(p[fbin(c,'value',val)]))
        res.append((k1, c, val, pm, nll))
    return res
def lay_doc(v, c, lval, age):
    v[0] = 1.0; v[1] = math.log1p(age)/10; v[2] = zsc(c,'value',lval)
fac3['k'] = 0
fac_idx = {}
for s_, grp_s in fac3.groupby('stay_id'):
    hrs_ = [b_[0] for b_ in bundles_B[s_]]
    for _, r in grp_s.iterrows():
        k1 = int(np.argmin(np.abs(np.array(hrs_) - r['hour'])))
        fac_idx[(s_, k1, r['event'])] = (r['predicted_mean'], r['nll'])
def cmp_head(type_fn, status_idx, ln_layout, stays, nsamp=32):
    d_pm, d_nll = [], []
    for s_ in stays:
        for (k1, c, val, pm, nll) in head_probs(s_, None, status_idx, ln_layout, nsamp) if False else []:
            pass
    return None
res_t = []
for c in ['glucose_finger','map_arterial']:
    for ti in range(15):
        d_pm = []
        for s_ in list(bundles_B.keys())[:4]:
            for (k1, cc, val, pm, nll) in head_probs(s_, ti, 0, lay_doc, nsamp=32, codes=(c,)):
                if (s_, k1, cc) in fac_idx: d_pm.append(abs(pm - fac_idx[(s_, k1, cc)][0]))
        res_t.append((round(float(np.median(d_pm)),3), c, ti))
    print(sorted([r for r in res_t if r[1] == c])[:3])
```