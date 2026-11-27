`vocab.json` has no categorical vocabularies (units, route, status and so on). An exact local re-implementation of the input side would therefore stay guesswork.

I am switching to a **head-level edit**:
- Add a correction to the value head's logits, driven by a dose readout from the state, which the last run showed is partly present.
- A head-only edit leaves every factual state unchanged. So I can compute the edited model's factual NLL exactly offline, once my local copy of the heads matches the tool's factual output.
- Only the rollout behaviour needs the tool.

This run does two things:
1. Records the plan in the notebook.
2. Estimates the **true one-step dose coefficients** from the training records. I regress the next finger reading on the glucose history, with flexible tier controls, plus insulin at lags 1 to 6, and do the same for MAP on norepinephrine.

Expected result: a clearly negative lag-1 insulin coefficient, near my earlier impulse estimate of about −2 per u/h, with small increments at longer lags. For norepinephrine, a positive lag-1 coefficient near +3.4 per mcg/kg/min.

```python
with open('/work/notes.md','a') as f: f.write("""
## Runs 39-43 (after restart 1)
- Old local-reimplementation function sources are lost (no getsource). VOCAB has no categorical vocabularies -> exact re-implementation is guesswork.
- States (fwd_2, sel1): x = [out 0:192 | h 4x192 192:960 | zpost 960:992]. Within-tier insulin dose is decodable from state: CV R2 out .54, h .67, all .68.
- NEW PLAN: head-level edit (value head + transition table), driven by a dose readout from the state. A head-only edit leaves factual states unchanged
  -> factual NLL can be computed offline once the local head matches the factual tool output. Correction must use dose minus protocol-expected dose
  (otherwise it double-counts the average effect already absorbed by the model).
""")
G=Gtr
def lagmat(df,t,L): return np.column_stack([df[t+1-l].values for l in range(1,L+1)])
rowsI=[]
for t in range(2,24):
    y=G['gf'][t+1].values; g0=G['gf'][t].values; g1=G['gf'][t-1].values; gl0=G['gl'][t].values
    dl=np.nan_to_num(lagmat(G['ins'],t,6)); ok=np.isfinite(y)&np.isfinite(g0)
    rowsI.append(pd.DataFrame(dict(y=y[ok],g0=g0[ok],g1=g1[ok],gl0=gl0[ok],t=t,**{f'd{l}':dl[ok,l-1] for l in range(1,7)})))
RI=pd.concat(rowsI,ignore_index=True)
def design_glu(R):
    T0=np.eye(5)[tier_of(R.g0.values)]; g0=(R.g0.values-200)/50
    m1=np.isfinite(R.g1.values); g1=np.where(m1,(R.g1.values-200)/50,0); T1=np.eye(5)[tier_of(np.where(m1,R.g1.values,100))]*m1[:,None]
    ml=np.isfinite(R.gl0.values); gl=np.where(ml,(R.gl0.values-200)/50,0)
    return np.column_stack([T0,T0*g0[:,None],T0*g0[:,None]**2,T1,g1,m1,gl,ml])
Xc=design_glu(RI); Xd=RI[[f'd{l}' for l in range(1,7)]].values
b=np.linalg.lstsq(np.column_stack([Xc,Xd]),RI.y.values,rcond=None)[0]
print("rows",len(RI),"true one-step insulin coefs lag1..6:",np.round(b[-6:],3))
```