The manual step loop is fast (0.87 ms per step). It matches the tool's states within float32 noise; the 0.23 maximum comes from the few hard bundles noted earlier.

Next is the Jacobian experiment on insulin drug bundles in the 100 training stays. For each such bundle I'll add ±0.5 along each of the 32 z directions. I'll take one step and read the predicted finger and lab glucose and arterial and cuff MAP at the next hourly reading. This tells me which z direction lowers glucose without moving MAP.

A consistent gradient across bundles, with glucose sensitivity well separated from MAP sensitivity, would support the z-injection design. Inconsistent or tiny gradients would refute it.

```python
regs = {}
for (s_,c_),d in train[train.stay_id.isin(set(S3.tolist())) & train.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].groupby(['stay_id','event']):
    regs[(s_,c_)] = (d.hour.values, d.value.values)
insh = {s_: d.hour.values for s_,d in ins_tr.groupby('stay_id')}
def reg_at(s_, c_, h_):
    if (s_,c_) not in regs: return np.nan, np.nan
    hh, vv = regs[(s_,c_)]; w_ = np.where(hh < h_-1e-6)[0]
    return (vv[w_[-1]], h_-hh[w_[-1]]) if len(w_) else (np.nan, np.nan)
def head_pred(ctx_, codes, lastv, lastage_h, col0):
    N = ctx_.shape[0]; mu, ls = zprior2(ctx_)
    view = ctx_ + lin2(mu,'z_head')
    q = F_.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in codes]); has = ~np.isnan(lastv)
    zl = np.array([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(codes,lastv)])
    ag = np.where(has, np.log1p(np.nan_to_num(lastage_h))/10.0, 0.0)
    feats = np.zeros((N,8)); feats[:,1] = has; feats[:,3] = ag; feats[:,2] = zl; feats[:,0] = col0
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(codes,lastv)])
    wc = q + T64('embed.code.weight')[cid] + torch.tensor(feats) @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[0]
    lg = gelu_(lin2(wc,'attr_heads.value_bin.0')) @ T64('attr_heads.value_bin.2.weight').T + T64('attr_heads.value_bin.2.bias') + T64('embed_transition.value_bin.weight')[cid*17+lb]
    p = torch.softmax(lg,-1); cen = torch.stack([CENT2[c] for c in codes])
    return (p[:,1:]*cen).sum(1).numpy()
CODES4 = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
def preds_after(s_, h_next, o_, h_, dt_s):
    cx = ctx_of_state(o_.double()[None], h_.double()[None], torch.tensor([dt_s],dtype=torch.float64))
    cx4 = cx.repeat(4,1)
    lv = []; la = []
    for c_ in CODES4:
        v_, a_ = reg_at(s_, c_, h_next); lv.append(v_); la.append(a_)
    hi = float(s_ in insh and np.any(insh[s_] < h_next))
    col0 = np.array([hi, hi, 0.0, 0.0])
    return head_pred(cx4, CODES4, np.array(lv), np.array(la), col0)
stays_ = list(dict.fromkeys(S3.tolist()))
insb = [k for k in range(5155) if hi_[k] and S3[k+1]==S3[k]]
rng2 = np.random.default_rng(0); samp = rng2.choice(insb, size=min(300,len(insb)), replace=False)
eps_ = 0.5; JG = []; t0_ = time.time()
for k in samp:
    s_ = int(S3[k]); ks_ = list(np.where(S3==s_)[0]); i0 = ks_.index(k)
    hp_ = torch.tensor(X3[k-1,192:960].reshape(4,192), dtype=WTd) if i0>0 else torch.zeros(4,192,dtype=WTd)
    op_ = torch.tensor(X3[k-1,:192], dtype=WTd) if i0>0 else torch.zeros(192,dtype=WTd)
    dtk = dts[k]; hf = hp_*torch.exp(-dtk*torch.exp(WT['core.flow.a'])); cx = ctx_from(op_, hf, dtk); z0 = zpost_mean(cx, Et[k], Ut[k])
    idx_m = np.arange(max(ks_[0], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
    hn = hr_[k+1]; dtn = float(dts[k+1])
    rows = []
    for i in range(-1,32):
        zz_ = z0.clone()
        if i>=0: zz_[i] += eps_
        h1, o1 = core_step(hf, Et[k], Ut[k], zz_, Et[idx_m], ages, V_ if 'V_' in globals() else {'incl':1,'unit':'s','age':'both'})
        rows.append(preds_after(s_, hn, o1, h1, dtn))
    rows = np.array(rows); JG.append((rows[1:]-rows[0])/eps_)
JG = np.array(JG)
print('time', round(time.time()-t0_,1), 'JG shape', JG.shape)
mJ = JG.mean(0); sJ = JG.std(0)
print('mean dGF/dz', mJ[:,0].round(2)); print('mean dMA/dz', mJ[:,2].round(2))
print('consistency |mean|/sd gf', (np.abs(mJ[:,0])/(sJ[:,0]+1e-9)).round(2))
```