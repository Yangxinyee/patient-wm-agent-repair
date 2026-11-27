q_of is exact (identity error 0.0), so the earlier spread came from the last-reading part, not from q. I can now simulate prior z draws as δq = q_of(prev, dt, eps) − q_of(prev, dt).

Next is the main data build on the 1,500 train stays in G3, run in parallel with a time guard:
- **factual rows:** with_code for all four codes, last bin and two z-draw deviations per row;
- **counterfactual rows:** for each stay with the drug running from hour 12, the change in with_code under insulin ×2/×½ and NE ×2/×½ from hour 12, with recorded readings kept;
- **dose paths:** recorded insulin and NE rates per hour.

These are the inputs for the decoder. Its factual variance must be small (factual cost), its mean response to these exact edits must be fixed, and it must not respond to the other drug. If this works I expect roughly 80k factual rows and a few thousand edited rows per edit.

```python
import multiprocessing as mp, time, pickle
def dose_arr(df_, drug):
    a_=np.zeros(27); sub=df_[df_.event==drug]
    for hh,rr in zip(sub.hour.values, sub.rate.values): a_[int(np.floor(hh))+1]=rr
    return a_
def proc_stay(s):
    df_=G3[s]; B_=bundles_of_stay(df_,RU0); St_=run_stay(B_); R_=gen_rows_all(df_,St=St_)
    tb_=np.array([b_['t'] for b_ in B_]); rng=np.random.default_rng(int(s)%(2**31))
    hours_=np.array([r_[2] for r_ in R_]); X_=np.stack([r_[8] for r_ in R_]).astype(np.float32)
    DQ=np.zeros((len(R_),2,192),np.float32)
    for i_,r_ in enumerate(R_):
        k_=int(np.argmin(np.abs(tb_-r_[2]))); prev=St_[k_-1] if k_>0 else np.zeros(992); dt=(tb_[k_]-tb_[k_-1])*3600 if k_>0 else 1.0
        q0,_=q_of(prev,dt)
        for j_ in range(2): DQ[i_,j_]=q_of(prev,dt,rng.standard_normal(32))[0]-q0
    sel=hours_>=12.5; ED={}
    for drug in ['insulin','norepinephrine']:
        m_=(df_.event==drug)&(df_.hour>=12)
        if not m_.any(): continue
        for f_ in [2.0,0.5]:
            d2=df_.copy(); d2.loc[m_,['rate','dose']]*=f_
            St2=run_stay(bundles_of_stay(d2,RU0)); R2=gen_rows_all(d2,St=St2)
            ED[(drug,f_)]=np.stack([r_[8] for r_ in R2]).astype(np.float32)[sel]-X_[sel]
    return dict(s=s, code=[r_[0] for r_ in R_], hour=hours_, prev=np.array([r_[3] for r_ in R_]), cur=np.array([r_[4] for r_ in R_]),
                X=X_, DQ=DQ, sel=sel, ED=ED, ins=dose_arr(df_,'insulin'), ne=dose_arr(df_,'norepinephrine'))
stays_all=list(G3.keys()); t0=time.time(); GEN=[]
ctx_mp=mp.get_context('fork')
with ctx_mp.Pool(8) as pool:
    for res_ in pool.imap_unordered(proc_stay, stays_all, chunksize=4):
        GEN.append(res_)
        if time.time()-t0>780: pool.terminate(); break
print('stays done', len(GEN), 'time', round(time.time()-t0,1))
pickle.dump(GEN, open('/work/gen_main.pkl','wb'))
print('rows', sum(len(g_['code']) for g_ in GEN), 'edits', collections.Counter(k_ for g_ in GEN for k_ in g_['ED']).most_common())
```