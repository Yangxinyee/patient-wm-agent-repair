Found the bug: `stay_data` reads a module-level DataFrame named `B`, and my run-293 line `B,rows,EU=stay_data(s)` overwrote it with a list — breaking every later call. `/work/panel.pkl` is a tuple, so it likely holds that frame and I can restore it.

Next run restores `B` from the panel and then inspects the `EU` layout, using only `zz_`-prefixed names so nothing else gets clobbered.

Confirming outcome: `stay_data` works again and I can see which `EU` entry is the action token.

```python
import numpy as np, pickle
zz_p=pickle.load(open('/work/panel.pkl','rb'))
print('panel len',len(zz_p))
for zz_j,zz_x in enumerate(zz_p):
    print(' [%d] %s %s has_stay_id=%s'%(zz_j,type(zz_x).__name__,getattr(zz_x,'shape',None),hasattr(zz_x,'stay_id')))
    if hasattr(zz_x,'columns'): print('     cols',list(zz_x.columns)[:12])
zz_cand=[zz_x for zz_x in zz_p if hasattr(zz_x,'stay_id')]
if zz_cand:
    B=zz_cand[0]; print('restored B:',type(B).__name__,getattr(B,'shape',None))
    try:
        zz_s=int(selS[0]); zz_B,zz_rws,zz_EU=stay_data(zz_s)
        print('stay_data OK stay',zz_s,'len EU',len(zz_EU),'rows[:10]',zz_rws[:10])
        for zz_k in range(min(8,len(zz_EU))):
            zz_e=zz_EU[zz_k]; print('EU[%d] %s len=%s'%(zz_k,type(zz_e).__name__,len(zz_e) if hasattr(zz_e,'__len__') else '-'))
            if isinstance(zz_e,(tuple,list)):
                for zz_i,zz_v in enumerate(zz_e):
                    zz_sh=getattr(zz_v,'shape',None)
                    if zz_sh is None: print('   [%d] %s %s'%(zz_i,type(zz_v).__name__,repr(zz_v)[:40])); continue
                    zz_V=np.asarray(zz_v.detach().cpu().numpy() if hasattr(zz_v,'detach') else zz_v,dtype=np.float64)
                    zz_ex=''
                    if zz_V.shape==(192,): zz_ex=' |v|=%.1f pi=%.2f pn=%.2f'%(np.linalg.norm(zz_V),float(pi.predict(zz_V.reshape(1,-1))[0]),float(pn.predict(zz_V.reshape(1,-1))[0]))
                    elif zz_V.ndim==2 and zz_V.shape[-1]==192: zz_ex=' pi=%s'%np.round(pi.predict(zz_V),2)[:4]
                    print('   [%d] %s %s%s'%(zz_i,type(zz_v).__name__,tuple(zz_sh),zz_ex))
    except Exception as zz_err: print('stay_data still broken:',type(zz_err).__name__,zz_err)
else: print('no frame with stay_id in panel')
```