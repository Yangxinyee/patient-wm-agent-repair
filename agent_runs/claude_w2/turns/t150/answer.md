Best so far: gap scaled by /10 (as in ctx) and type_emb row 1, giving RMSE 7.4 against the tool. That is still well above the ~2–3 reached before, so the `last_numeric` assignment matters. Next is a coordinate-descent search over the 8 columns: each column takes one of has / no-has / log-age (s or h) / z_last / 0 / 1. I'll also try the status options, with type 1 and gap/10 fixed. If the head is otherwise right, the error should fall to the z-noise floor (a few mg/dL, mean near 0).

```python
base1=head_base(Xo[p_],Xh[p_],dt_t,W,10.0)
cands_=['has','nohas','la_s','la_h','zl','zero','one']
def eval_cfg(cols,stat,ty_=1):
    st=torch.as_tensor({'none':np.zeros(len(fa4)),'has':has_,'nohas':1-has_}[stat].astype(np.int64))
    P=head_probs_mc(base1,cod,f8_make(cols),lbt,st,W,ty=ty_,nz=0); rm,me,_=pm_score(P); return rm,me
cur=['has','la_s','zl','zero','zero','zero','zero','zero']; cur_s='none'; best_rm=eval_cfg(cur,cur_s)[0]
for rnd in range(3):
    for j in range(8):
        for c_ in cands_:
            tr_=cur.copy(); tr_[j]=c_; rm,me=eval_cfg(tr_,cur_s)
            if rm<best_rm-1e-4: best_rm, cur = rm, tr_
    for s_ in ['none','has','nohas']:
        rm,me=eval_cfg(cur,s_)
        if rm<best_rm-1e-4: best_rm,cur_s=rm,s_
    print(rnd, round(best_rm,3), cur, cur_s, eval_cfg(cur,cur_s))
```