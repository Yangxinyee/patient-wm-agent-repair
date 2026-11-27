The instrumented loop runs 100 stays (5,369 bundles) in 0.4 s, so collecting internals is cheap. Each atom carries event, value, rate, dose, kind and minute.

Next I check how `kind` encodes starts versus END atoms. Then I tabulate, per prelude head, the slot queries' attention to the e, u and z tokens by bundle type, giving the mean, sd and the 5th percentile. I am looking for a head with weight ≈1 and near-zero spread on u at every insulin bundle. I also need to know what that head attends at reading and other bundles, because the rate decoder must output ≈0 there.

```python
at2=ds2['at']; print(at2.groupby(['event','kind']).size().to_string())
bd2=ds2['bd']; N2=len(bd2)
def flag(cond):
    f_=np.zeros(N2,bool); f_[at2.loc[cond,'row'].values]=True; return f_
ins_s=flag((at2.event=='insulin')&at2.rate.notna()); ne_s=flag((at2.event=='norepinephrine')&at2.rate.notna())
endf=flag(at2.event.isin(['insulin','norepinephrine'])&at2.rate.isna())
gl=flag(at2.event.isin(['glucose_finger','glucose_lab'])); mp=flag(at2.event.isin(['map_arterial','map_cuff']))
aw2=ST2['aw'][:,:,:4,:].mean(2).numpy()   # N,4heads,7 tokens (slot-query mean)
types={'ins':ins_s&~ne_s,'ins+ne':ins_s&ne_s,'ne':ne_s&~ins_s,'end_only':endf&~ins_s&~ne_s,'gluc_read':gl,'map_only':mp&~gl,'other':~(ins_s|ne_s|endf|gl|mp)}
for nm_,m_ in types.items():
    if m_.sum()==0: continue
    s_=f"{nm_:10s} n={m_.sum():5d} "
    for h_ in range(4):
        s_+=f"| h{h_} e {aw2[m_,h_,4].mean():.2f}±{aw2[m_,h_,4].std():.2f} u {aw2[m_,h_,5].mean():.2f}±{aw2[m_,h_,5].std():.2f} (p5 {np.percentile(aw2[m_,h_,5],5):.2f}) z {aw2[m_,h_,6].mean():.2f} "
    print(s_)
```