The dtype fix worked: the original head tensors are float64 and the head now runs. The new failure is in my summary table. This pandas version leaves the grouping columns out of `groupby.apply`, so `g.code` does not exist.

I will rewrite the summary as an explicit loop over (code, drug) groups and rerun the two offline evaluations. As before, I expect near-zero cost and shift on gated-off glucose rows, and a total cost well under the ungated 0.018.

```python
def eval_edit(E,label='',verbose=True):
    parts=[]
    for i in range(5):
        Xs_,R_=datasets[i]; o=head_run(Xs_,R_,E); pmb,nllb,_,_=base_out[i]
        parts.append(pd.DataFrame({'sh':npv(o[0])-pmb,'dn':npv(o[1])-nllb}))
    D_=pd.concat(parts,ignore_index=True)
    for c_ in ['code','drug','dev1','nedev1','stay']: D_[c_]=A_[c_].values
    out=[]
    for (c,dr),g in D_.groupby(['code','drug']):
        xc='dev1' if c in (6,10) else 'nedev1'; x=g[xc].values; y=g.sh.values; k_=np.isfinite(x)
        sl=np.cov(x[k_],y[k_])[0,1]/np.var(x[k_],ddof=1)
        out.append(dict(code=c,drug=dr,n=len(g),dNLL=g.dn.mean(),sh_mean=g.sh.mean(),sh_sd=g.sh.std(),slope=sl))
    if verbose: print(label,'overall dNLL %.4f'%D_.dn.mean()); print(pd.DataFrame(out).round(4).to_string(index=False))
    return D_
E0g=gated_edit(0.0,0.0); _=eval_edit(E0g,'a=0 (units removed):')
Eg=gated_edit(-1.9,3.0); Dg=eval_edit(Eg,'gated a_ins=-1.9 a_ne=3:')
```