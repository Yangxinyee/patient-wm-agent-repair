The inventory shows that runs 177–188 built `inj.py` (a replica step with a custom z function, `bstep_z`; a teacher-forced runner, `tf_stay_z`). They also computed:

- a PCA of the action encoding u (`U_`, `S_`, `Vt_`, `npc`, `Pc_`);
- policy fits (`pi_i`, `pi_n`) and residual weights (`w_ins`, `w_ne`);
- Jacobians of how a perturbation of the z token propagates through the state (`JAC`, `Jlag`, `J1m`);
- a state-noise covariance and decoders (`Sv`, `Sinv`, `DEC1`, `DB1`, `dbest`, `rel`).

That matches the plan: refresh a dose memory through the z channel at every step. The results of those last runs were never delivered.

Next I will:
- print the `inj.py` code;
- print the shapes and scalar values of those variables;
- check in ARCHITECTURE.md whether the core uses the z posterior mean or a sample during rollouts.

That last point decides whether I must also pin the z log-variance outputs. I'll keep the output compact. If the decoder variables show good signal-to-noise for reading the injected memory back from the context, the memory loop is feasible. If not, I'll fall back to a lag-1-only gain edit.

```python
import numpy as np, inspect
print(open('/work/inj.py').read())
names=['npc','Pc_','Zc','w_ins','w_ne','pi_i','pi_n','JAC','Jlag','Js','Jm','J1m','DEC1','DB1','dbest','rel','sig','noise','Sv','Sinv','Mq','w_','V_','dec','r2','fit','cf','grid','Ug','res_tf','idx','cur','seen']
for n in names:
    v=globals().get(n)
    if isinstance(v,np.ndarray): print(n,'arr',v.shape, (np.round(v.ravel()[:4],3) if v.size<=4 or v.ndim==1 and v.size<10 else ''))
    elif isinstance(v,(int,float,np.floating)): print(n,'=',np.round(v,4))
    elif isinstance(v,dict): print(n,'dict',list(v.keys())[:8])
    elif isinstance(v,(list,tuple)): print(n,type(v).__name__,len(v))
    else: print(n,type(v).__name__)
A_=open('/model/ARCHITECTURE.md').read()
for kw in ['sample','posterior','rollout','z_post','logvar']:
    for m in re.finditer(kw,A_): print(kw,'|',A_[max(0,m.start()-80):m.start()+120].replace('\n',' ')); break
```