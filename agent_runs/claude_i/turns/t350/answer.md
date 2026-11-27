The current edit fails the factual check: value NLL rose by 0.037 nats, against a limit of 0.005. It hurt glucose (+0.046 on finger glucose) and MAP (+0.053 on arterial MAP). So as it stands it cannot be submitted.

I have 11 Python runs left, so each run has to count. This run will:
- read the result of forward pass 5, which I believe was the dose_edit run for this same edit;
- work out where the factual damage comes from (see the checks below);
- recall how the edit is built;
- if the attention change touches only the value half of the weight, save a version at half strength, to test how factual loss and dose response scale with strength.

The factual checks:
- whether the damage sits in readings where the edit moves the predicted mean, or also in readings it leaves alone (a leak);
- the slope of the original model's error on the edit's shift. A slope near 1 means the injected signal is real and about the right size. A slope near 0 or below means it is noise or has the wrong sign.

```python
import os, json, glob, re, numpy as np, pandas as pd
def sh(x,n=300):
    s=str(x); return s if len(s)<=n else s[:n]+'...'
note=[]
try:
    for k in [5,6]: print(k, sorted(os.listdir(f'/work/fwd_{k}')))
    for p in sorted(glob.glob('/work/fwd_5/*')):
        f=os.path.basename(p)
        if f.endswith('.json'): print(f, sh(open(p).read(),500))
        if f.startswith('dose_edit') and f.endswith('.csv'):
            df=pd.read_csv(p); print(f, df.shape, sh(list(df.columns),250))
            cat=[c for c in df.columns if df[c].dtype==object and df[c].nunique()<=10]
            num=[c for c in df.columns if df[c].dtype!=object and 'stay' not in c.lower()]
            if cat:
                t=df.groupby(cat[0])[num].mean().round(3); print(t.to_string()[:900]); note.append(f+' '+t.to_string()[:400].replace('\n',' | '))
except Exception as ex: print('E1',ex)
try:
    fo=pd.read_csv('/work/fwd_6/factual_original.csv'); fe=pd.read_csv('/work/fwd_6/factual_edited.csv')
    cols=list(fo.columns); print('fcols',cols)
    pick=lambda keys,ex=(): [c for c in cols if any(k in c.lower() for k in keys) and not any(e in c.lower() for e in ex)]
    mc=pick(['mean'])[0]; nc=pick(['nll'])[0]; ec=pick(['event'])[0]
    vcs=pick(['value'],['mean','nll','pred','bin']); vc=vcs[0] if vcs else None
    tcs=pick(['hour','time']); tc=tcs[0] if tcs else None
    dm=fe[mc]-fo[mc]; dn=fe[nc]-fo[nc]
    for ev,g in fo.groupby(ec):
        i=g.index; a=dm[i]; n_=dn[i]; big=a.abs()>0.5
        s=f'{ev}: dNLL={n_.mean():.4f} f_big={big.mean():.2f} dNLL_big={n_[big].mean():.4f} dNLL_small={n_[~big].mean():.4f} dm_mu={a.mean():.2f} dm_sd={a.std():.2f}'
        if vc:
            r=fo.loc[i,vc]-fo.loc[i,mc]; b=(a*r).sum()/max((a*a).sum(),1e-9); s+=f' b={b:.2f}'
        if tc: s+=' dm_t='+str(a.groupby(pd.cut(fo.loc[i,tc],4)).mean().round(2).values.tolist())
        print(s); note.append(s)
except Exception as ex: print('E2',ex)
try:
    for n in edited():
        a=np.asarray(load(n),dtype=np.float64); b=np.asarray(load_original(n),dtype=np.float64); d=a-b
        if d.ndim==2:
            rr=np.where(np.abs(d).sum(1)>0)[0]; cc=np.where(np.abs(d).sum(0)>0)[0]
            sv=np.linalg.svd(d,compute_uv=False)[:3]
            print(n, d.shape,'rows',len(rr),rr[:4].tolist(),rr.max() if len(rr) else -1,'cols',len(cc),cc[:6].tolist(),'sv',np.round(sv,3))
        else:
            ii=np.where(d!=0)[0]; print(n, d.shape,'nz',len(ii),ii[:6].tolist(),'max',round(float(np.abs(d).max()),3))
    ep=np.load('/work/edit_parts.npz'); print('edit_parts',{k:(ep[k].shape if ep[k].ndim else float(ep[k])) for k in ep.files})
except Exception as ex: print('E3',ex)
try:
    for p in sorted(glob.glob('/work/*.py')):
        s=open(p).read(); print(os.path.basename(p), re.findall(r'^def (\w+)', s, re.M)[:15])
    fn=[k for k,v in globals().items() if callable(v) and getattr(v,'__module__',None)=='__main__' and not k.startswith('_')]
    print('funcs', sh(fn,600))
except Exception as ex: print('E4',ex)
try:
    n='core.prelude.0.attn.kv.weight'; W0=np.asarray(load_original(n),dtype=np.float32); W1=np.asarray(load(n),dtype=np.float32)
    DW_full=W1-W0; np.save('/work/DW_kv_full.npy',DW_full)
    D=W0.shape[1]; rr=np.where(np.abs(DW_full).sum(1)>0)[0]
    print('kv shape',W0.shape,'changed rows in K half',(rr<D).sum(),'V half',(rr>=D).sum())
    if W0.shape[0]==2*D and (rr<D).sum()==0:
        save_edit(n, W0+0.5*DW_full); print('SAVED kv at 0.5x'); note.append('saved kv 0.5x of DW_full (/work/DW_kv_full.npy)')
except Exception as ex: print('E5',ex)
with open('/work/notes.md','a') as f:
    f.write('\n## Session 8 (restart after 348 replies; 6 fwd, 290 runs)\n- fwd6 factual of current edit: +0.037 NLL (FAILS). '+' ; '.join(note)[:1500]+'\n')
```