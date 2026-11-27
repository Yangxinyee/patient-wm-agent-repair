The decoder change came out as zero (slope -0.004). That is a bug, not a finding: bundle times are in hours (-2, -1, 0), while I compared them with reading times in minutes. Every row therefore used the stay's final state. I will redo the comparison with times in hours and dt in seconds. The question is still whether the lag-1 decoder moves by about 0.8× the dose change under insulin x2.

```python
def q_rows(df_, code='glucose_finger', hmin=13):
    B_=bundles_of_stay(df_,RU0); S_=run_stay(B_); T_=np.array([b['t'] for b in B_],dtype=float)
    rr=df_[(df_.event==code)&(df_.hour>=hmin)]; Q=[]; H=[]
    for hr_ in rr.hour.values:
        j=np.where(T_<hr_-1e-6)[0][-1]; Q.append(np.asarray(q_of(S_[j], (hr_-T_[j])*3600.0)[0]).ravel()); H.append(hr_)
    return np.array(Q), np.array(H), T_
dlt=[]; dd_=[]; QA_=[]
for s in selT[:40]:
    df_=G3[s]; df2=df_.copy(); mk=(df2.event=='insulin')&(df2.hour>=12); df2.loc[mk,'rate']*=2; df2.loc[mk,'dose']*=2
    Qa,Ha,Ta=q_rows(df_); Qb,Hb,_=q_rows(df2)
    if len(Qa)==0: continue
    da_=dose_arr(df_,'insulin'); dlt.append((Qb-Qa)); dd_.append(np.array([[da_[int(h)-k+1] for k in (1,2,3)] for h in Ha])); QA_.append(Qa)
dQ=np.concatenate(dlt); dv=np.concatenate(dd_)
print('T sample',np.round(Ta[:6],3))
for k in range(3): print('dec lag',k+1,'mean d',round((dQ@WD[k]).mean(),3),'mean dose',round(dv[:,k].mean(),3),'slope',round(np.polyfit(dv[:,k],dQ@WD[k],1)[0],3))
```