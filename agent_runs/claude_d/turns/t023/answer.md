The finger-glucose measurement rule is simple and ignorable. After any insulin hour, glucose_finger is measured the next hour every time (100%). Otherwise it is measured at random 30% of the time, regardless of the last value, time since the last reading or norepinephrine. Missingness therefore depends only on recorded insulin, so weighting non-insulin hours by 1/0.3 removes the selection problem in the multi-hour estimates.

Next I will run sequential g-estimation of the insulin impulse response β_k, the effect of 1 u/h at hour s on glucose_finger at s+k with all other doses held fixed, for k = 1 to 12:
- **Instrument:** the dose deviation from its policy mean given base and source;
- **Outcome:** the outcome with later doses' estimated effects subtracted;
- **Weights:** inverse probability of measurement;
- **Controls:** glucose and base at s.

If β_k stays near −2 at every lag (full persistence), the x2 effect accumulates roughly linearly. If β_k decays, the effect is short-lived and much smaller.

```python
# Build arrays per stay x hour (hours -1..24 -> index 0..25)
stays = c.index.get_level_values('stay_id').unique()
H = 26
def arr(col): return c[col].values.reshape(len(stays), H)
GF = arr('gf'); GL = arr('gl'); INS = arr('ins0'); NOR = arr('nor0')
Gdec = np.where(~np.isnan(GF), GF, GL)
BASE = step_of(np.nan_to_num(Gdec, nan=-1))
SRC = np.where(~np.isnan(GF), 0, np.where(~np.isnan(GL),1,2))
GIVEN = INS>0
# policy mean given base, source, given
EPS = np.zeros_like(INS)
for b in [1,2,3,4,6]:
    for sr in [0,1]:
        msk = GIVEN & (BASE==b) & (SRC==sr)
        if msk.sum()>0: EPS[msk] = INS[msk] - INS[msk].mean()
print('eps var', EPS[GIVEN].var(), 'given n', GIVEN.sum(), 'given w/o base', (GIVEN&(BASE==0)).sum())
# measurement weight for gf at t: 1 if INS[t-1]>0 else 1/0.3
W = np.ones_like(GF); W[:,1:] = np.where(INS[:,:-1]>0, 1.0, 1/0.3); W[:,0]=1/0.5
K=12; beta=np.zeros(K+1)
for k in range(1,K+1):
    rows=[]
    Yt=[];E=[];A=[];C1=[];C2=[];Wt=[]
    for t in range(k, H):
        s=t-k
        obs = ~np.isnan(GF[:,t])
        yt = GF[:,t] - sum(beta[j]*INS[:,t-j] for j in range(1,k))
        sel = obs & GIVEN[:,s] & ~np.isnan(Gdec[:,s])
        Yt.append(yt[sel]); E.append(EPS[sel,s]); A.append(INS[sel,s]); C1.append(Gdec[sel,s]); C2.append(BASE[sel,s]); Wt.append(W[sel,t])
    Yt=np.concatenate(Yt);E=np.concatenate(E);A=np.concatenate(A);C1=np.concatenate(C1);C2=np.concatenate(C2);Wt=np.concatenate(Wt)
    # residualize Y on controls (weighted), then IV ratio
    Xc = np.c_[np.ones_like(C1), C1, C1**2/100, (C2[:,None]==np.array([1,2,3,4,6])).astype(float)]
    sw=np.sqrt(Wt); coef=np.linalg.lstsq(Xc*sw[:,None], Yt*sw, rcond=None)[0]; r = Yt - Xc@coef
    beta[k] = np.sum(Wt*r*E)/np.sum(Wt*A*E)
    # se approx
    u = r - beta[k]*(A-A.mean())
    se = np.sqrt(np.sum((Wt*E*u)**2))/abs(np.sum(Wt*A*E))
    print(k, len(Yt), 'beta', round(beta[k],3), 'se', round(se,3))
```