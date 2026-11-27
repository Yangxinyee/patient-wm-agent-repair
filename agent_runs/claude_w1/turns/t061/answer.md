The dose_edit file has per stay-hour shifts, a factual baseline and a "reached" flag. I'll tabulate by hour, for reached rows only:
- the original model's mean shift for all four edits;
- my record-based truth: the deconvolved impulse response times the extra dose from hour 12 on.

That shows whether the model's small effect is constant or builds up over hours, which tells me how much the model carries its own glucose shifts forward.

```python
hI_full=np.array([-2.08,-2.41,-2.16,-1.79,-1.26,-0.95,-0.77,-0.63,-0.55,-0.43,-0.33,-0.30])
hN_full=np.array([3.40,2.60,1.62,0.94,0.41,0.33])
def truth_rows(de,edit,drug,hh,factor):
    out=[]
    for s_,t in zip(de.stay_id,de.hour):
        tot=0.0
        for k in range(1,len(hh)+1):
            tk=t-k
            if tk<12: break
            dval=Ptu[drug].get((s_,tk),np.nan)
            if dval==dval: tot+=hh[k-1]*(factor-1)*dval
        out.append(tot)
    return np.array(out)
tab_=[]
for edit,drug,hh,fac in [('insulin_x2','insulin',hI_full,2.0),('insulin_x0.5','insulin',hI_full,0.5),('norepinephrine_x2','norepinephrine',hN_full,2.0),('norepinephrine_x0.5','norepinephrine',hN_full,0.5)]:
    d_=de1[(de1.edit==edit)&(de1.reached)].copy()
    if len(d_)==0: print(edit,'none; edits=',de1.edit.unique()); continue
    d_['truth']=truth_rows(d_,edit,drug,hh,fac)
    g_=d_.groupby('hour')[['shift','truth']].mean().round(2)
    print(edit,'pooled model',round(d_['shift'].mean(),3),'truth',round(d_.truth.mean(),3),'n',len(d_))
    print(g_.T.to_string())
```