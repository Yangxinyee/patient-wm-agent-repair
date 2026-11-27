Run 288 found two things:

- **fwd_14 (factual pass for the current edit, v9):** value NLL went from 1.4938 to 1.7282 (+0.234). Lowering the channel injection and raising the b_in gain made the factual fit worse, not better, so my hypothesis about the injection size is refuted.
- **fwd_15 (states):** the edit barely moves the state at drug bundles or just after them (median |Δout| about 0.05 against a norm of 13.9). But the perturbation grows the longer a stay goes without a drug bundle. Before a stay's first drug bundle it is as large as the state itself (median |Δh| about 1700 against a norm of about 1600).

**Conclusion:** the damage is a leak or feedback loop that only shows up where no drug token is being read, probably at the start of stays. It is not the intended dose path.

**Next run (information only, short output):**
- the added dose effect by hour for each edit in fwd_10 and fwd_11;
- the NLL change from fwd_12, fwd_13 and fwd_14, split into readings before and after a stay's first drug bundle;
- the helper functions still in the kernel;
- a short excerpt of how bundle tokens are built.

If most of the NLL cost falls in the pre-drug readings, the leak hypothesis is confirmed. The fix would then be to zero the head-1 readout on the tokens seen in those bundles.

```python
import numpy as np, pandas as pd, re, torch
S=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); S.append(s)
for k in (10, 11):
    do=pd.read_csv(f'/work/fwd_{k}/dose_edit_original.csv'); de=pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv')
    if 'reached' in do.columns:
        do=do[do.reached.astype(bool)]; de=de[de.reached.astype(bool)]
    for e_ in sorted(do['edit'].astype(str).unique()):
        mo=do[do.edit.astype(str)==e_].groupby('hour')['shift'].mean(); me=de[de.edit.astype(str)==e_].groupby('hour')['shift'].mean()
        P(f'fwd{k} {e_} add/h:', ' '.join(f'{x:+.1f}' for x in (me-mo).values[:12]))
st=np.load('/work/fwd_15/states_original.npz')
sid=np.asarray(st['stay_id']); hr=np.asarray(st['hour']); dr=(np.asarray(st['has_insulin'])+np.asarray(st['has_norepinephrine']))>0
first=pd.DataFrame({'sid':sid,'hr':hr})[dr].groupby('sid')['hr'].min()
ss=set(sid.tolist())
for k in (12,13,14):
    fo=pd.read_csv(f'/work/fwd_{k}/factual_original.csv'); fe=pd.read_csv(f'/work/fwd_{k}/factual_edited.csv')
    ins=fo['stay_id'].isin(ss); fh=fo['stay_id'].map(first)
    pre=ins & (fh.isna() | (fo['hour']<fh))
    dn=fe['nll']-fo['nll']; dp=fe['predicted_mean']-fo['predicted_mean']
    P(f'fwd{k}: cover {ins.mean():.2f} dNLL {dn.mean():+.4f} = pre {dn[pre].sum()/len(dn):+.4f} (n{pre.sum()}) + post {dn[~pre].sum()/len(dn):+.4f}')
    for ev in sorted(fo['event'].astype(str).unique()):
        m=fo['event'].astype(str)==ev
        P(f'   {ev}: dNLL pre {dn[m&pre].mean():+.3f} post {dn[m&~pre].mean():+.3f} | dp_rms pre {np.sqrt((dp[m&pre]**2).mean()):.1f} post {np.sqrt((dp[m&~pre]**2).mean()):.1f}')
fn=[k_ for k_,v_ in globals().items() if callable(v_) and not k_.startswith('_') and getattr(v_,'__module__','')=='__main__' and not isinstance(v_,type)]
print('FUNCS:', ' '.join(fn)[:900])
ar=[f'{k_}{tuple(v_.shape)}' for k_,v_ in globals().items() if not k_.startswith('_') and isinstance(v_,(np.ndarray,torch.Tensor)) and v_.ndim>=1 and v_.size if isinstance(v_,np.ndarray) else (isinstance(v_,torch.Tensor) and not k_.startswith('_') and v_.ndim>=1)]
print('ARRAYS:', ' '.join(ar)[:900])
txt=open('/model/ARCHITECTURE.md').read().split('\n')
hits=[l.strip()[:150] for l in txt if re.search(r'token|u_|drug|infus|first|init', l, re.I)]
print('ARCH:', ' || '.join(hits)[:1800])
with open('/work/notes.md','a') as f:
    f.write('\n- run289: fwd_14 = factual v9: NLL +0.234 (worse than v7 s=.3 +.173). fwd_15 states v9: perturbation tiny at drug/lag1 bundles, grows with lag, HUGE before first drug (|dh|~|h|) -> leak/feedback where no drug token.\n' + '\n'.join('- '+s for s in S) + '\n')
```