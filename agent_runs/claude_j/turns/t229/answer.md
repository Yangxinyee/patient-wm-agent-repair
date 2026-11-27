The factual distortion is mostly extrapolation:
- RMS 21–23 mg/dL at hours ≤ 6 and on rows with no prior insulin, states the counterfactual rows never cover;
- 3.5–5.6 on drug-exposed rows.

A useful trick follows from GELU: gelu(x) − gelu(−x) = x exactly. So two units fed ±r plus a shared gate G output exactly 2r when G is large, and 0 when G is very negative. That gives a gated linear readout with no constant to cancel. The gate can switch off both MAP codes and unexposed states.

Whether exposure gating is enough depends on the exposed factual rows: does the readout's output there hurt NLL, or does it partly capture the true dose-deviation effect the model under-uses? Next I split the factual NLL change by exposure and by hour group.
- If exposed rows show roughly neutral or negative change, gating on exposure solves the factual problem.
- If they also cost a lot, I need a better readout.

```python
def dnll_groups(ix,oof,v,groups):
    o=[]
    for nm,g in groups:
        r=np.where(g)[0]; P0=pnew(ix[r],np.zeros(len(r)),v); P1=pnew(ix[r],oof[r],v); b=BIN[ix[r]]; a=np.arange(len(r))
        o.append('%s n%d %.4f'%(nm,len(r),(-np.log(P1[a,b])+np.log(P0[a,b])).mean()))
    return ' | '.join(o)
for wf in (1,5):
    ix,oof,mn=fit_eval(isG,vG,100,wf=wf); f=NONE[ix]; h=M.hour.values[ix]; st=M.stay.values[ix]
    ex=np.array([h_>first[s_] for h_,s_ in zip(h,st)])
    print('wf',wf,dnll_groups(ix,oof,vG,(('exp',f&ex),('exp>12',f&ex&(h>12)),('unexp',f&~ex))))
```