Run 289 failed on a syntax error in one list comprehension, so nothing ran and nothing was written to the notebook. I am repeating the same information-gathering run with that line fixed and the output kept short:
- the added dose effect by hour for each edit in fwd_10 and fwd_11;
- the NLL change in fwd_12 to fwd_14, split into readings before and after a stay's first drug bundle;
- the bundle sizes before the first drug bundle;
- the helper functions and arrays still in the kernel;
- a short excerpt from ARCHITECTURE.md on how bundle tokens are built.

If most of the NLL cost sits in the pre-drug readings, the leak hypothesis is confirmed.

```python
import numpy as np, pandas as pd, re, torch
S=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
def rmask(df):
    return df['reached'].astype(str).str.lower().isin(['true','1','1.0'])
for k in (10, 11):
    do=pd.read_csv(f'/work/fwd_{k}/dose_edit_original.csv'); de=pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv')
    if 'reached' in do.columns:
        do=do[rmask(do)]; de=de[rmask(de)]
    for e_ in sorted(do['edit'].astype(str).unique()):
        mo=do[do['edit'].astype(str)==e_].groupby('hour')['shift'].mean()
        me=de[de['edit'].astype(str)==e_].groupby('hour')['shift'].mean()
        P(f'fwd{k} {e_} add/h:', ' '.join(f'{x:+.1f}' for x in (me-mo).values[:12]))
st=np.load('/work/fwd_15/states_original.npz')
sid=np.asarray(st['stay_id']); hr=np.asarray(st['hour'])
dr=np.asarray(st['has_insulin']).astype(bool) | np.asarray(st['has_norepinephrine']).astype(bool)
na=np.asarray(st['n_atoms'])
first=pd.DataFrame({'sid':sid,'hr':hr})[dr].groupby('sid')['hr'].min()
P(f'stays {len(set(sid.tolist()))}, with drug {len(first)}; n_atoms mean all {na.mean():.2f}')
ss=set(sid.tolist())
for k in (12,13,14):
    fo=pd.read_csv(f'/work/fwd_{k}/factual_original.csv'); fe=pd.read_csv(f'/work/fwd_{k}/factual_edited.csv')
    ins=fo['stay_id'].isin(ss); fh=fo['stay_id'].map(first)
    pre=ins & (fh.isna() | (fo['hour']<fh))
    dn=fe['nll']-fo['nll']; dp=fe['predicted_mean']-fo['predicted_mean']
    P(f'fwd{k}: cover {ins.mean():.2f} dNLL {dn.mean():+.4f} = pre {dn[pre].sum()/len(dn):+.4f} (n{int(pre.sum())}) + post {dn[~pre].sum()/len(dn):+.4f}')
    for ev in sorted(fo['event'].astype(str).unique()):
        m=fo['event'].astype(str)==ev
        P(f'   {ev}: dNLL pre {dn[m&pre].mean():+.3f} post {dn[m&~pre].mean():+.3f} | dp_rms pre {np.sqrt((dp[m&pre]**2).mean()):.1f} post {np.sqrt((dp[m&~pre]**2).mean()):.1f}')
fn=[k_ for k_,v_ in list(globals().items()) if callable(v_) and not k_.startswith('_') and getattr(v_,'__module__','')=='__main__' and not isinstance(v_,type)]
print('FUNCS:', ' '.join(fn)[:900])
ar=[]
for k_,v_ in list(globals().items()):
    if k_.startswith('_'): continue
    if isinstance(v_,np.ndarray) and v_.ndim>=1: ar.append(f'{k_}{v_.shape}')
    elif isinstance(v_,torch.Tensor) and v_.ndim>=1: ar.append(f'{k_}{tuple(v_.shape)}')
print('ARRAYS:', ' '.join(ar)[:900])
txt=open('/model/ARCHITECTURE.md').read().split('\n')
hits=[l.strip()[:140] for l in txt if re.search(r'token|u_const|first|init', l, re.I)]
print('ARCH:', ' || '.join(hits)[:1200])
with open('/work/notes.md','a') as f:
    f.write('\n- run289 was a syntax error (lost). run290: fwd_14 = factual v9: NLL +0.234. fwd_15 states v9: perturbation tiny at drug/lag1, grows with lag, HUGE before first drug (|dh|~|h|).\n' + '\n'.join('- '+s for s in S) + '\n')
```