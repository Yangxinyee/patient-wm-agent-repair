The architecture refresher clarifies three things:

- **Where the injected signal ends up.** The heads read `out` (plus decayed h, the time gap and z) through `query_norm`. The glucose head also leans heavily on its own last‑value register.
- **Why v3 did not help.** It pushed a rate signal into the prelude's head‑1 value path. That signal then passes through `prelude_norm`, the recurrent block and the coda before it reaches `out`.
- **What fwd_6 showed.** All four responses got slightly *smaller* (insulin ×2 −0.145 vs −0.348, NE ×½ −0.163 vs −0.245). So the injection either barely reaches the glucose prediction or reaches it with the wrong sign.

fwd_7 has teacher‑forced states for the original and v3 models on the same stays. Next I will measure how large the v3 change in `out`, h and z is at insulin, NE and other bundles. I will then translate it into predicted glucose with a linear map from `out` to the model's own predicted next finger glucose, fitted on fwd_2/fwd_3.

- If the out‑change at insulin bundles is tiny, or implies a positive glucose shift, then v3 was simply mis‑aimed. I would rebuild the injection with a direction calibrated on `out` → predicted glucose.
- If it is large and in the right direction, then the problem lies downstream (persistence or rollout), not in the injection.

```python
so=np.load('/work/fwd_7/states_original.npz'); se=np.load('/work/fwd_7/states_edited.npz')
xo=so['x']; xe=se['x']; hi=so['has_insulin'].astype(bool); hn=so['has_norepinephrine'].astype(bool)
d7=xe-xo
for nm,sl in [('out',slice(0,192)),('h',slice(192,960)),('z',slice(960,992))]:
    nb=np.linalg.norm(d7[:,sl],axis=1); no=np.linalg.norm(xo[:,sl],axis=1)
    print(nm,'rel ins %.4f ne %.4f none %.4f'%(nb[hi].mean()/no[hi].mean(),nb[hn&~hi].mean()/no[hn&~hi].mean(),nb[~hi&~hn].mean()/no[~hi&~hn].mean()))
fc=pd.read_csv('/work/fwd_2/factual_original.csv'); print(fc.columns.tolist()); print(fc.head(3).to_string())
print(so['stay_id'][:5],so['hour'][:5],so['n_atoms'][:5])
```