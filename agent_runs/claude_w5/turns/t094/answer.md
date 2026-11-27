Run 89 is uninformative. The constant-offset fit diverged (the fitted vector reached norm about 220 and the error went up to about 20), so the regularisation was far too weak for such an ill-conditioned problem.

I'll go back to concrete structural hypotheses for the gap term. My gap vector is large (norm about 15, comparable to out), and leaving it out helps, so maybe I add it in the wrong place. Two natural variants:
- the gap vector is added to the slot mean before the LayerNorm, so it gets normalised: ctx = out + LN(mean h + gap);
- the whole sum is normalised: ctx = LN(out + mean h + gap).

I'll test each with several input scalings. A variant that drops the error far below 0.6 would explain the earlier failures.

```python
def G_in(dt, mode):
    x_ = {'s10': math.log(dt)/10, 'min10': math.log(dt/60)/10, 'h': math.log(dt/3600), 'h10': math.log(dt/3600)/10, 's': math.log(dt)}[mode]
    return lin_(Fnn.gelu(lin_(torch.tensor([x_]), 'gap.0')), 'gap.2')
dts_ = []
for s_ in list(bundles_B.keys())[:20]:
    bl_ = bundles_B[s_]
    for k_ in range(1, len(bl_)):
        atoms = bl_[k_][1]
        wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
        if aa or not wa: continue
        dts_.append((bl_[k_][0]-bl_[k_-1][0])*3600.0)
        if len(dts_) >= 150: break
    if len(dts_) >= 150: break
HFm_ = torch.stack([flow_c(Xt[0,192:960].reshape(4,192), 1.0, 0) for _ in range(1)])
raw_hf = []
for s_ in list(bundles_B.keys())[:20]:
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
    for k_ in range(1, len(bl_)):
        atoms = bl_[k_][1]
        wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
        if aa or not wa: continue
        dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
        raw_hf.append(flow_c(Xt[r0_+k_-1,192:960].reshape(4,192), dt, 1.0).mean(0))
        if len(raw_hf) >= 150: break
    if len(raw_hf) >= 150: break
MH_ = torch.stack(raw_hf)
wN, bN = T_['core.out_norm.weight'], T_['core.out_norm.bias']
def lnx(x): return Fnn.layer_norm(x, (192,), wN, bN, 1e-5)
for gm in ['s10','min10','h','h10']:
    GG = torch.stack([G_in(d_, gm) for d_ in dts_])
    v1 = batch_err(O_ + lnx(MH_ + GG)); v3 = batch_err(lnx(O_ + MH_ + GG)); v2 = batch_err(O_ + lnx(MH_) + GG)
    v4 = batch_err(O_ + lnx(MH_) + lnx(GG))
    print(gm, 'out+LN(mh+g)', round(v1,3), 'LN(out+mh+g)', round(v3,3), 'doc', round(v2,3), 'out+LN(mh)+LN(g)', round(v4,3))
```