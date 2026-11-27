`build_ds` gives a self-contained dataset:
- atoms and bundles;
- per-bundle e/u encodings, dt and time in seconds;
- per-stay bundle rows;
- the readings table;
- a head-input dict R (previous row, dt, code, has-last, age, z of last value, last value bin, noise, observed bin).

`tf_eval` runs `tf_states_ds` and then `head_value`, so any edited parameter dict can be scored offline on the factual NLL directly.

Next I write an instrumented copy of the `tf_states_ds` loop. It uses all prior bundles as cross-attention memory, ages computed from `tsec`, and an all-true mask, matching the disassembly of `tf_states`. To collect internals it calls `step_b(..., ret=True)`. First I check what `ret=True` returns: the keys and shapes of tokens, norm1 output, attention weights, E, hn and y. I also confirm that my copy reproduces `tf_states_ds` exactly; if the max difference is tiny, I can trust it for fitting decoders.

```python
def tf_loop(P_, ds, keep=None):
    bd=ds['bd']; N=len(bd); srows=ds['srows']; e_all=ds['e']; u_all=ds['u']
    Hn=torch.zeros(N,4,192); On=torch.zeros(N,192); store={}
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
            if keep is None: y_,o_=res_[0],res_[1]
            else:
                y_,o_=res_['y'],res_['out']
                for kk_ in keep:
                    v_=res_[kk_]
                    if kk_ not in store: store[kk_]=torch.zeros((N,)+tuple(v_.shape[1:]))
                    if tuple(store[kk_].shape[1:])!=tuple(v_.shape[1:]): store[kk_]=None; continue
                    store[kk_][rows_k]=v_
            Hn[rows_k]=y_; On[rows_k]=o_
    return Hn,On,store
# inspect ret structure on first two steps
srows_=ds_['srows']; rows0=np.array([r_[1] for r_ in srows_]); pr0=np.array([r_[0] for r_ in srows_])
Hs0,Os0=tf_states_ds(P,ds_)[:2]
hprev=Hs0[pr0]; oprev=Os0[pr0]; dt=ds_['dts'][rows0]
cx=ctx_fn(oprev,hprev,dt,P); z=zpost_mean(cx,ds_['e'][rows0],ds_['u'][rows0],P)
mem=torch.tensor(np.stack([r_[:2] for r_ in srows_])); t_=ds_['tsec'][mem]; ages=(t_[:,-1:]-t_).clamp(0)
res_=step_b(hprev,ds_['e'][rows0],ds_['u'][rows0],z,dt,ds_['e'][mem],ages,torch.ones(len(rows0),2).bool(),P,ret=True)
print(type(res_).__name__)
if isinstance(res_,dict):
    for k_,v_ in res_.items(): print(' ',k_, tuple(v_.shape) if hasattr(v_,'shape') else type(v_).__name__)
else:
    for i_,v_ in enumerate(res_): print(' ',i_, tuple(v_.shape) if hasattr(v_,'shape') else type(v_).__name__)
H1,O1,_=tf_loop(P,ds_)
print('replica check maxdiff H', float((H1-Hs0).abs().max()), 'O', float((O1-Os0).abs().max()))
```