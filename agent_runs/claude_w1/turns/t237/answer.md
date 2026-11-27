A bundle holds its time 't', its e/u sets and labels, and run_stay returns the 992-number state after each bundle. Now the direct test. For 40 train stays I will run the exact replica on the recorded record and on an insulin-x2-from-hour-12 record. For each finger reading at hours 13-24 I compute the head query from the preceding state, then compare the ridge lag-1 decoder's change with the actual lag-1 dose change.

If the decoder change is about 0.8-0.9 of the dose change, the earlier 16% figure was an artefact of how EDR was built. If it is again about 0.16, the state really does represent large dose edits only weakly.

```python
WD=np.asarray(DEC['glucose_finger'].coef_).reshape(3,192)
def q_rows(df_, code='glucose_finger', hmin=13):
    B_=bundles_of_stay(df_,RU0); S_=run_stay(B_); T_=np.array([b['t'] for b in B_])
    rr=df_[(df_.event==code)&(df_.hour>=hmin)]
    Q=[]; H=[]
    for hr_ in rr.hour.values:
        tr=round(hr_*60); j=np.where(T_<tr)[0][-1]; Q.append(q_of(S_[j], (tr-T_[j])*60.0)); H.append(hr_)
    return np.array(Q), np.array(H), T_
dlt=[]; dd_=[]
for s in selT[:40]:
    df_=G3[s]; df2=df_.copy(); mk=(df2.event=='insulin')&(df2.hour>=12)
    df2.loc[mk,'rate']*=2; df2.loc[mk,'dose']*=2
    Qa,Ha,Ta=q_rows(df_); Qb,Hb,_=q_rows(df2)
    if len(Qa)==0: continue
    da_=dose_arr(df_,'insulin')
    dlt.append((Qb-Qa)@WD[0]); dd_.append(np.array([da_[int(h)-1+1] if False else da_[int(h)] for h in Ha]))
print('Tunits sample',Ta[:4])
dl=np.concatenate(dlt); dv=np.concatenate(dd_)
print('n',len(dl),'mean dDec',round(dl.mean(),3),'mean dose col(h)',round(dv.mean(),3))
```