Run 89 gives the key quantities:
- **Decodability:** the head-time query q decodes the current insulin rate with R2 0.69 and its deviation from the band mean with R2 0.41. The hidden activations do about the same.
- **The model's existing response:** its predicted glucose has a positive partial slope on the dose deviation (+0.31). It uses the dose as a proxy for glucose level and does not treat it as a cause; the records give −0.9 here (noisy; the large-sample estimate is −2.0).
- **Policy timing:** insulin never runs in the first 2 hours.

A simple "noisy linear dose feature" model fits both R2 values: the state holds a dose signal f = r + e with noise sd ≈0.9 u/h, plus a clean band signal. Under that model, a head term c·(f − band mean) with c = −2.0 − 0.31 gives the right interventional slope. Its factual cost is only about +0.0004 nats per glucose reading. A probe fitted to the deviation would need its output scaled by 1/0.41.

To apply such a term only to glucose_finger rows, I need gating in the shared head. So I'll compute head inputs and hidden activations for every value row in fwd_3, all four codes. Then I'll look for dead hidden units, which would allow clean code gating, and measure how well a linear readout on the activations can be zero for the other codes.

```python
rec = recs.sort_values(['stay_id','hour'])
LASTREG = {}
for code in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    rc_ = rec[rec.event==code][['stay_id','hour','value']].copy()
    rc_['ph'] = rc_.groupby('stay_id')['hour'].shift(1); rc_['pv'] = rc_.groupby('stay_id')['value'].shift(1)
    for s,h,ph,pv in zip(rc_.stay_id, rc_.hour, rc_.ph, rc_.pv): LASTREG[(code,s,round(h,4))] = (ph,pv)
bidx = {}
for s in np.unique(SST): w_ = np.where(SST==s)[0]; bidx[s] = (w_, SHR[w_])
def head_feats(s, hour, code, zmode='mean'):
    w_, hr = bidx[s]; k = np.searchsorted(hr, hour-1e-6, side='left')-1
    if k < 0: return None
    i0 = w_[k]; out_ = torch.tensor(SX[i0,:192]); h_ = torch.tensor(SX[i0,192:960]).reshape(4,192)
    dt = (hour - SHR[i0])*3600.0; ctx = HD.ctx(out_, h_, dt); mu, ls = HD.zprior(ctx)
    view = ctx + HD.L(mu,'z_head'); q = HD.LN(view + W['type_emb.weight'][1],'query_norm')
    ph, pv = LASTREG.get((code,s,round(hour,4)), (np.nan,np.nan)); num = torch.zeros(8); lb = 0
    if pv==pv:
        lb = fbin(code,'value',float(pv)); num[0]=1; num[1]=math.log(1+max(SHR[i0]-ph,0)*60)/10; num[2]=zsc(code,'value',float(pv)); num[3]=1
    x = q + W['embed.code.weight'][CODE[code]] + HD.L(num,'last_numeric') + W['embed.bins.value.weight'][lb]
    return i0, q.numpy(), HD.hidden(x).numpy(), CODE[code]*17+lb
FA = []; meta = []
for r_ in fac.itertuples():
    o = head_feats(r_.stay_id, r_.hour, r_.event)
    if o is None: continue
    FA.append(o[2]); meta.append((r_.stay_id, r_.hour, r_.event, o[0], o[3], r_.observed, r_.predicted_mean))
FA = np.array(FA); MT = pd.DataFrame(meta, columns=['stay_id','hour','event','i0','trow','obs','pm'])
print(FA.shape, MT.event.value_counts().to_dict())
mx = FA.max(0); mean_by = {c: FA[MT.event.values==c].mean(0) for c in MT.event.unique()}
print('units with max act < 0.05:', int((mx<0.05).sum()), ' <0.2:', int((mx<0.2).sum()))
W2 = W['attr_heads.value_bin.2.weight'].numpy(); usage = np.abs(W2).sum(0)*FA.std(0)
print('lowest-usage units (idx, max act, |W2|col*sd):', [(int(i), round(float(mx[i]),3), round(float(usage[i]),4)) for i in np.argsort(usage)[:8]])
```