Run 241 shows strong persistence. Shifting the glucose readings from hour 13 on by −13 mg/dL lowered the replica's next glucose_finger predictions by 9.9 on average. It also exposed a units slip: `age` is in seconds, so the head's last-value input was not shifted, and the full response is probably even larger.

So the model does track its recent readings. The small counterfactual shift in run 240 must therefore come from an interaction: when doses are doubled, something pulls the prediction back up.

Next I will redo the test with ages in seconds and decompose it in one pass. `amp_eval` on the shifted stay returns both "shifted readings, recorded doses" and "shifted readings, doubled doses". On the original stay it returns "original" and "doubled doses only".

If shifted readings alone give about −12 but shifted plus doubled give about −2, the doubled dose itself cancels about 10 mg/dL of the lower readings. That would point to a learned association with the opposite sign, which a correct edit must remove. If both conditions give similar shifts, I would have to look for the explanation elsewhere.

```python
def shifted(s,dv=-13.0,m0=780):
    B=[(bm,[(a[0],a[1],a[2]._replace(value=a[2].value+dv)) if (a[0].startswith('glucose') and bm>=m0) else a for a in at]) for bm,at in BUN[s]]
    rt=RTs[s].copy(); lm=rt.minute-rt.age/60.0
    msk=rt.event.str.startswith('glucose')&(lm>=m0)&(rt.has_last>0); rt.loc[msk,'last_val']=rt.loc[msk,'last_val']+dv
    return B,rt,msk.sum()
R={'sh':[],'shx2':[],'x2':[]}
for s in st30:
    B0,R0=BUN[s],RTs[s]; Bm,Rm,nm=shifted(s)
    BUN[s],RTs[s]=Bm,Rm; CACHE.clear(); a1=amp_eval(1.0,[s])[0]; BUN[s],RTs[s]=B0,R0; CACHE.clear(); a0=amp_eval(1.0,[s])[0]
    ev=R0.event.values; mi=R0.minute.values; c=CEN['glucose_finger']; mm=lambda L:torch.softmax(L,-1).mean(1).numpy()[:,1:]@c
    sel=(ev=='glucose_finger')&(mi>=840)&(mi<1500); base_=mm(a0[3])[sel]
    R['sh']+=list(mm(a1[3])[sel]-base_); R['shx2']+=list(mm(a1[4])[sel]-base_); R['x2']+=list(mm(a0[4])[sel]-base_)
print({k:'%.2f'%np.mean(v) for k,v in R.items()}, len(R['sh']))
```