With `ret=True`, `step_b` exposes E, hf, hn and the prelude attention weights `aw`, shaped (batch, 4 heads, 7 query tokens, 7 key tokens). On this first reading bundle, head 1 puts all its weight on token 5 and head 3 on token 4; heads 0 and 2 spread over the slot tokens. Token 4 is probably e and token 5 probably u. My notes said head 3 reads u at insulin bundles, so attention probably switches by bundle type. That has to be measured before I pick the head that carries the dose.

Next I run the instrumented loop over the 100 training stays in sel2 (about 5,400 bundles). I'll tabulate each head's mean attention from the slot queries to each token, split by bundle content: insulin start, NE start, infusion END only, glucose reading, MAP-only reading. The test that matters: some head should put weight ≈1.0 on the u token at every insulin bundle, with essentially zero spread. That head becomes my exact channel for the rate. If none does, the staging write will need a different route.

```python
def tf_loop(P_, ds, keep=None):
    bd=ds['bd']; N=len(bd); srows=ds['srows']; e_all=ds['e']; u_all=ds['u']
    Hn=torch.zeros(N,4,192); On=torch.zeros(N,192); store={}; Zs=torch.zeros(N,32)
    L_=max(len(r_) for r_ in srows)
    with torch.no_grad():
        for k_ in range(L_):
            idx=[i_ for i_,r_ in enumerate(srows) if len(r_)>k_]
            rows_k=np.array([srows[i_][k_] for i_ in idx])
            if k_==0:
                hprev=torch.zeros(len(rows_k),4,192); oprev=torch.zeros(len(rows_k),192); dt=torch.ones(len(rows_k))
            else:
                pr=np.array([srows[i_][k_-1] for i_ in idx]); hprev=Hn[pr]; oprev=On[pr]; dt=ds['dts'][rows_k]
            cx=ctx_fn(oprev,hprev,dt,P_); z=zpost_mean(cx,e_all[rows_k],u_all[rows_k],P_)
            mem=torch.tensor(np.stack([srows[i_][:k_+1] for i_ in idx]))
            t_=ds['tsec'][mem]; ages=(t_[:,-1:]-t_).clamp(0)
            res_=step_b(hprev,e_all[rows_k],u_all[rows_k],z,dt,e_all[mem],ages,torch.ones(len(rows_k),k_+1).bool(),P_,ret=(keep is not None))
            y_,o_=res_[0],res_[1]; Zs[rows_k]=z
            if keep is not None:
                for kk_ in keep:
                    v_=res_[2][kk_]
                    if kk_ not in store: store[kk_]=torch.zeros((N,)+tuple(v_.shape[1:]))
                    store[kk_][rows_k]=v_
            Hn[rows_k]=y_; On[rows_k]=o_
    store['z']=Zs
    return Hn,On,store
print('at cols', list(ds_['at'].columns)); print(ds_['at'].head(6).to_string()[:900])
ds2=build_ds(tr, sel2, 0)
t0_=time.time(); H2,O2,ST2=tf_loop(P,ds2,keep=['aw','E','hf','hn']); print('loop s', round(time.time()-t0_,1), 'N', len(ds2['bd']))
```