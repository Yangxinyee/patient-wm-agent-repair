The run failed on my own bookkeeping line: I searched for `def run_rollout(` after renaming it to `run_rollout_h`. Nothing else ran. I fix that and rerun the same propagation experiment. It shifts the drawn glucose_finger at hour 13 by +2 bins and tracks the predicted-mean response at hours 14–24. A flat response means full propagation; a decay gives the model's reversion rate.

```python
src=open('/work/rollout.py').read()
src=src.replace("def run_rollout(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None,store=None,store_h=False):",
                "def run_rollout_h(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None,store=None,store_h=False,hook_P=None,hook_draw=None,hook_state=None):")
src=src.replace("            P=P/4\n","            P=P/4\n            if hook_P is not None: P=hook_P(k,hn_,hj,P,dict(seq_si=seq_si,seq_arm=seq_arm,seq_p=seq_p,drawn=drawn,base=base,inv=inv,out=out,h=h,un=un))\n")
src=src.replace("                drawn[hn_[hd],A['loc'].values[hj[hd]]]=b\n","                drawn[hn_[hd],A['loc'].values[hj[hd]]]=b\n                if hook_draw is not None: hook_draw(k,hn_[hd],hj[hd],drawn,dict(seq_si=seq_si,seq_arm=seq_arm,seq_p=seq_p))\n")
src=src.replace("        h[act_n]=y; out[act_n]=LN(y.mean(1),W,'core.out_norm')\n","        h[act_n]=y; out[act_n]=LN(y.mean(1),W,'core.out_norm')\n        if hook_state is not None: hook_state(k,act_n,h,out,dict(seq_si=seq_si,seq_arm=seq_arm,seq_p=seq_p,e=e,u=u,Ep=Ep,hn=hn))\n")
hsrc=src[src.index('def run_rollout_h('):]; hsrc=hsrc[:hsrc.index('def _z_vec')]
assert 'hook_P(k' in hsrc and 'hook_draw(k' in hsrc and 'hook_state(k' in hsrc
open('/work/rollout_h.py','w').write("import numpy as np, torch, torch.nn.functional as F, math, pandas as pd\n"+hsrc); exec(open('/work/rollout_h.py').read())
ids_p=list(rng_.choice(cand_ids,100,replace=False)); Rp=prep_rollout(tr,ids_p); Ap=Rp['A']
gf13=set(np.where((Ap.code.values==CODE['glucose_finger'])&(np.abs(Ap.t.values-13)<1e-6))[0])
def hd_shift(k,hn,hj,drawn,ctx):
    a=ctx['seq_arm'][hn]; sel=(a==1)&np.array([j in gf13 for j in hj])
    if sel.any():
        loc=Ap['loc'].values[hj[sel]]; drawn[hn[sel],loc]=np.minimum(drawn[hn[sel],loc]+2,16)
t0_=time.time(); resp=run_rollout_h(Rp,W,[{},{}],n_paths=20,seed=1,hook_draw=hd_shift); print('%.1fs'%(time.time()-t0_))
rows,reached,o=readout(Rp,resp,[{},{}],lo=13,hi=24)
hrs_=np.round(Ap.t.values[rows]).astype(int); sid_=Ap.si.values[rows]
has13=np.isin(sid_,Ap.si.values[list(gf13)])
dif=np.nanmean(o[1]-o[0],1)
for hh_ in range(13,25):
    m_=(hrs_==hh_)&has13; print(hh_, m_.sum(), np.round(np.nanmean(dif[m_]),2))
```