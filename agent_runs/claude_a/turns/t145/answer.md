The 192-dim channel decoder helps, but the trade-off remains. At equal weighting it gives residual 0.28 at insulin bundles and about 0.1 off-target. Pushing the off-target residual down to 0.03 raises the insulin residual to 0.47. Rate information is not linearly separable from the e and slot tokens within head 3's view alone.

One lever is left before I need a nonlinear gate, which costs a second LayerNorm division. The heads attend differently by bundle type. Head 1 reads the u token at drug bundles, and at reading bundles it reads the empty u token, which is a constant. Heads 2 and 3 both read e at reading bundles. So one repurposed value channel in each of heads 1–3 gives a joint linear decoder that can difference out the e-token garbage while keeping the rate.

I'll fit that joint 576-feature decoder on sel2 and check it on 100 fresh training stays to rule out overfitting. Success would be a held-out insulin residual of ≲0.12 with off-target residual ≲0.08. If it fails, I move to the FFN gate.

```python
rng_v=np.random.default_rng(7)
cand_v=np.setdiff1d(ftr.stay_id.values if 'stay_id' in ftr.columns else ftr.index.values, np.array(sel2))
selv=list(rng_v.choice(cand_v,100,replace=False))
dsv=build_ds(tr, selv, 0); Hv,Ov,STv=tf_loop(P,dsv,keep=['aw','hf'])
def feats(ds_,ST_):
    N_=len(ds_['bd']); T_,n1_,_=prelude_parts(P, ST_['hf'], ds_['e'], ds_['u'], ST_['z'])
    aw_=ST_['aw']; Fh=[torch.einsum('bqk,bkd->bqd', aw_[:,h_,:4,:], n1_).mean(1) for h_ in range(4)]
    at_=ds_['at']; bd_=ds_['bd']
    def fl(c_):
        f_=np.zeros(N_,bool); f_[at_.loc[c_,'row'].values]=True; return f_
    s0=at_.kind==0
    fl_={'ins':fl((at_.event=='insulin')&s0),'ne':fl((at_.event=='norepinephrine')&s0),'gl':fl(at_.event.isin(['glucose_finger','glucose_lab'])),
         'mp':fl(at_.event.isin(['map_arterial','map_cuff'])),'end':fl(at_.event.isin(['insulin','norepinephrine'])&(at_.kind==2))}
    r_=np.zeros(N_); a_=at_[(at_.event=='insulin')&s0]; r_[a_.row.values]=a_.rate.values
    return torch.cat(Fh,1).numpy(), fl_, r_
Ftr,fltr,rtr=feats(ds2,ST2); Fva,flva,rva=feats(dsv,STv)
def report(p_,y_,fl_,lab):
    drug_=fl_['ins']|fl_['ne']
    s_=f"{lab:28s} ins {np.std(p_[fl_['ins']]-y_[fl_['ins']]):.3f} ne_only {np.std(p_[fl_['ne']&~fl_['ins']]):.3f}"
    for nm_,m_ in [('gl',fl_['gl']),('map_only',fl_['mp']&~fl_['gl']),('end_only',fl_['end']&~drug_),('other',~(drug_|fl_['gl']|fl_['mp']|fl_['end']))]:
        s_+=f" | {nm_} {np.mean(p_[m_]-y_[m_]):+.3f}±{np.std(p_[m_]-y_[m_]):.3f}"
    print(s_)
for cols_,lab_ in [(slice(576,768),'h3'),(slice(192,768),'h1h2h3'),(slice(0,768),'h0-h3')]:
    for wo_ in [1.0,10.0]:
        Xa=np.c_[Ftr[:,cols_],np.ones(len(Ftr))]; wt=np.where(fltr['ins']|fltr['ne'],1.0,wo_); sw=np.sqrt(wt)
        w_,_,_,_=np.linalg.lstsq(Xa*sw[:,None], rtr*sw, rcond=None)
        report(Xa@w_, rtr, fltr, f"{lab_} woff={wo_} TRAIN")
        report(np.c_[Fva[:,cols_],np.ones(len(Fva))]@w_, rva, flva, f"{lab_} woff={wo_} VALID")
```