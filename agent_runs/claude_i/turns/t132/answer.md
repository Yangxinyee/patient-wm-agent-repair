Two details differ from the architecture text. `last_numeric` takes 8 inputs, not 4, and there are 15 type embeddings. The likely layout of the 8 inputs is [has_last, log(1+age)/10] followed by the same six numeric slots the atom embedding uses (value, has value, dose, has dose, rate, has rate). The value head is a 192→192→17 MLP, plus a 221 = 13×17 transition table.

Next I will implement the value head on top of the tool's exact states (fwd_3) and predict each glucose_finger reading in fwd_2. I compare my predicted means with the tool's for every type row and two layouts of the 8 inputs, using the prior-mean z for a deterministic check.

What I expect: one combination gives a small error. It should be about 1–2 mg/dL mean absolute difference, the size of the tool's own 4-draw noise. That would show the head is reproduced. Large errors everywhere would mean another detail, such as the age unit or the absent-bin embeddings, is wrong.

```python
CENT = {k: np.asarray(v, float) for k, v in VOCAB['value_bin_centres'].items()}
print('centre keys', list(CENT.keys())[:6], 'len', len(next(iter(CENT.values()))))
bmap = {(int(S3[i_]), int(round(float(min_[i_])))): i_ for i_ in range(NB)}
def view_ctx(k_prev, dt_next):
    hf = HHt[k_prev] * torch.exp(-dt_next*torch.exp(WT['core.flow.a']))
    g_ = linw(Fnn.gelu(linw(torch.log(torch.clamp(dt_next, min=1.0)).reshape(1)/10., 'gap.0')), 'gap.2')
    return OUTt[k_prev] + LNw(hf.mean(0), 'core.out_norm') + g_
def zprior(ctx):
    p_ = linw(Fnn.gelu(linw(ctx, 'z_prior.0')), 'z_prior.2'); return p_[..., :32], p_[..., 32:]
F2s = F2[F2.stay_id.isin(set(S3.tolist()))].sort_values(['stay_id','hour']).copy()
rows_ = []
for (sid_, ev_), g_ in F2s.groupby(['stay_id','event']):
    prevv = np.r_[np.nan, g_.observed.values[:-1]]; preva = np.r_[np.nan, np.diff(g_.hour.values)*3600.]
    for (h_, obs_, pm_, pv_, pa_) in zip(g_.hour.values, g_.observed.values, g_.predicted_mean.values, prevv, preva):
        kb = bmap.get((int(sid_), int(round(h_*60))))
        if kb is None or kb == 0 or not prev_ok[kb]: continue
        rows_.append((kb, ev_, obs_, pm_, pv_, pa_))
print('readings matched', len(rows_))
def pred_mean(r_, typ_, lay, zmode='mean'):
    kb, ev_, obs_, pm_, pv_, pa_ = r_
    ctx = view_ctx(kb-1, dts[kb]); zm_, zls_ = zprior(ctx)
    view = ctx + linw(zm_, 'z_head')
    q = LNw(view + WT['type_emb.weight'][typ_], 'query_norm')
    has = 0. if np.isnan(pv_) else 1.
    zl = 0. if not has else float(zsc(ev_, 'value', pv_)); ag = 0. if not has else np.log1p(pa_)/10.
    feats = {'A': [has, ag, zl, has, 0, 0, 0, 0], 'B': [has, ag, zl, 0, 0, 0, 0, 0]}[lay]
    lb = 0 if not has else int(fbin(ev_, 'value', np.array([pv_]))[0])
    wc = q + WT['embed.code.weight'][CID[ev_]] + torch.tensor(feats, dtype=torch.float32) @ WT['last_numeric.weight'].T + WT['last_numeric.bias'] \
         + WT['embed.bins.value.weight'][lb] + WT['embed.bins.rate.weight'][0] + WT['last_status.weight'][0]
    lg = linw(Fnn.gelu(linw(wc, 'attr_heads.value_bin.0')), 'attr_heads.value_bin.2') + WT['embed_transition.value_bin.weight'][CID[ev_]*17 + lb]
    p_ = torch.softmax(lg, -1).numpy()
    return float((p_[1:] * CENT[ev_][1:]).sum() / p_[1:].sum()) if len(CENT[ev_]) == 17 else float((p_[1:] * CENT[ev_]).sum()/p_[1:].sum())
sub_r = [r_ for r_ in rows_ if r_[1] == 'glucose_finger'][:150]
for lay in ['A','B']:
    mads_ = [np.mean([abs(pred_mean(r_, t_, lay) - r_[3]) for r_ in sub_r]) for t_ in range(15)]
    print(lay, 'best type', int(np.argmin(mads_)), 'MAD', np.round(sorted(mads_)[:3], 2))
```