The hooked engine lets me shift the head's predicted distribution at each hidden glucose or MAP reading inside the rollout, and record how the shift propagates through the drawn values. The earlier runs therefore probably measured which direct shift kernel reproduces the true effect. Those results are not in the notes, so they can only be in kernel variables. This run prints the small variables that look like kernels or oracle results, without recomputing anything. If they hold an oracle-calibrated insulin kernel and a propagation factor kap, I can go straight to building the edit in the weights. If not, I'll rerun a short oracle calibration.

```python
def show(nm,lim=400):
    if nm not in globals(): print(nm,'-- missing'); return
    v=globals()[nm]
    try:
        if isinstance(v,(np.ndarray,torch.Tensor)):
            a=np.asarray(v); s=f'{type(v).__name__} shape {a.shape}'
            if a.size<=40: s+=' '+np.array2string(a.ravel(),precision=3,max_line_width=250)
        elif isinstance(v,(dict,)): s='dict keys '+str(list(v.keys())[:15])
        elif isinstance(v,(list,tuple)): s=f'{type(v).__name__} len {len(v)} '+repr(v)[:lim]
        elif isinstance(v,(int,float,str,np.floating)): s=repr(v)[:lim]
        elif isinstance(v,pd.DataFrame): s='DF '+str(v.shape)+' cols '+str(list(v.columns)[:20])
        else: s=type(v).__name__
    except Exception as ex: s='err '+str(ex)
    print(nm,':',s[:lim])
for nm in ['K','Kf','Kins','Knor','Kv','Kv_i','Gins','Gne','Gi','Gn','G','Gf','kap','DGi_o','best','best_c','best_l','best_s','best_t','best_v','best_rm','sv_best','res_o','resC','resI','resS','res_c','res_t','res_v','res_s','res_f','res_l','score','opt','variants','eval_cfg','arms5','armsC','armsI','arms_o','ids30','idsC','idsI','idsI2','ids_o','cands','cand','tru','tool','mins_tool','both','pm_score','base_err','errs','errs_z','alpha','U_alpha','gscale','scl']:
    show(nm)
```