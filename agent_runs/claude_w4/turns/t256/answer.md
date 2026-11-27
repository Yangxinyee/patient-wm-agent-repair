The fwd_7 comparison shows the v3 edit is mis‑aimed. At insulin and NE bundles it barely changes the state: `out` moves by 0.08–0.10% and h by 0.1%. At bundles *without* drugs, though, `out` moves by 0.8% and h by 2.8%.

The likely reason: prelude head 1 reads the u token at every bundle. At non‑drug bundles that token is the constant u_const, and my rate readouts were fitted only at drug bundles, so they give a nonzero value on u_const. The edit is therefore mostly a constant push at every measurement bundle. That would explain why fwd_6's dose responses shrank instead of growing.

Any rebuilt carrier must have readouts that are exactly zero on LN1(u_const), and it probably needs a much larger gain at drug bundles.

To size that gain, I need a translation from a change in `out` (and slot‑mean h) into a change in predicted next finger glucose. I'll fit a ridge on the original states, using the bundle just before each reading, against the factual predicted mean from fwd_2. I'll then apply it to the v3 differences. If the implied glucose shift at insulin bundles is ≪1 mg/dL while non‑drug bundles show a systematic offset, the diagnosis is confirmed. I'll also look at the tuning.csv columns, so I can attach rates to bundles.

```python
from sklearn.linear_model import RidgeCV
sid=so['stay_id']; hr=so['hour']
idx=np.full(len(fc),-1)
for s_,g in fc.groupby('stay_id'):
    bi=np.where(sid==s_)[0]
    if len(bi)==0: continue
    hb=hr[bi]; assert np.all(np.diff(hb)>=0)
    pos=np.searchsorted(hb,g['hour'].values,side='left')-1
    ok=pos>=0; idx[g.index[ok]]=bi[pos[ok]]
fc['bi']=idx
def feat(x): return np.hstack([x[:,0:192], x[:,192:960].reshape(-1,4,192).mean(1)])
RG={}
for ev in ['glucose_finger','map_arterial']:
    m=(fc.event==ev)&(fc.bi>=0); g=fc[m]
    Xf_=feat(xo[g.bi.values]); y=g.predicted_mean.values
    r=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Xf_,y); RG[ev]=r
    dp=feat(d7[g.bi.values])@r.coef_
    b=g.bi.values
    print(ev,'R2 %.3f alpha %.3g'%(r.score(Xf_,y),r.alpha_),
          'dpred ins-bundle mean %.3f sd %.3f | ne-bundle %.3f | none-bundle mean %.3f sd %.3f'%(
          dp[hi[b]].mean(),dp[hi[b]].std(),dp[hn[b]&~hi[b]].mean(),dp[~hi[b]&~hn[b]].mean(),dp[~hi[b]&~hn[b]].std()))
tu=pd.read_csv('/data/tuning.csv',nrows=5); print(tu.columns.tolist()); print(tu.head(5).to_string())
```