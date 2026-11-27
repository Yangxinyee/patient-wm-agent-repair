All insulin records have the same units, and rates come in steps of 0.5 u/h with dose equal to rate. So no categorical column explains the pattern; the binary split between {1.0, 1.5} and the other rates must arise inside the model's own computation.

Next I compute my z for every insulin bundle of several stays, each from the previous stored state, and compare dimensions 9 and 20 with the actual values grouped by rate.
- If mine are near +1 for every rate, my implementation is missing a pathway the model uses, perhaps in how u_k or e_k is formed.
- If mine also split by rate but at a different place, the bins or z-scores are slightly off.

```python
def z_from_states(sid_, o=UB, end_obs=True):
    mm=S_['stay_id']==sid_; Xs_=S_['x'][mm]; evs_=tr[tr.stay_id==sid_].sort_values('hour'); bb=build_bundles(evs_,o)
    assert len(bb)==mm.sum()
    res_=[]
    for k in range(1,len(bb)):
        out_=torch.tensor(Xs_[k-1,:192]); h_=torch.tensor(Xs_[k-1,192:960]).reshape(4,192)
        dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
        ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
        at=bb[k][1]
        ob=[a for a in at if a['ek']!=2 or (end_obs and a['kind']==2)]; ac=[a for a in at if a['ek']==2 and not (end_obs and a['kind']==2)]
        e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
        u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
        zp=MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32].numpy()
        ir=[a['rate'][1] for a in ac if a['code']==8 and a['kind']==0]
        res_.append((sid_, bb[k][0], ir[0] if ir else np.nan, zp, Xs_[k,960:]))
    return res_
allr=[]
for sid_ in np.unique(S_['stay_id'])[:6]: allr+=z_from_states(sid_)
rr=np.array([t[2] for t in allr]); zp_=np.array([t[3] for t in allr]); za_=np.array([t[4] for t in allr])
for r0 in [0.5,1.0,1.5,2.0,2.5,3.0]:
    m_=np.isclose(rr,r0)
    if m_.any(): print(r0, m_.sum(), 'mine d9,d20', np.round(zp_[m_][:,[9,20]].mean(0),2), 'actual', np.round(za_[m_][:,[9,20]].mean(0),2))
print('overall median abs err', np.median(np.abs(zp_-za_).mean(1)))
```