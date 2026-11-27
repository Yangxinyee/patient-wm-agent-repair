The kernel ran out of memory: I materialised 300k×192 float64 arrays on top of a large panel DataFrame, and all variables are gone. My notes and the saved base panels are still in /work.

From now on I will use memory-light computations. Drug rates take few unique values (insulin in 0.5 steps, norepinephrine in 0.01 steps), so the embedding term D can be computed per unique rate with frequency weights instead of per atom.

Next I will
1. record the latest findings in the notes;
2. reload the helpers and weights;
3. redo the embedding-magnitude check in grouped form: per-code mean of D, within-code spread, the gap between codes, and the row norms of the categorical tables.

```python
import numpy as np, pandas as pd, json, os
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 29-36
- NE innovation does not predict next MAP in no-MAP hours (-0.22 se .55): doses carry no hidden info.
- Implied NE x2 readout via LP deconvolution (h_k) applied to tuning reached rows: arterial -0.31 (boot sd .15); cuff-only stays (independent) -0.37 (boot mean -0.28 sd .36).
  Pooled ~ -0.32 +- 0.14. Insulin same method: -0.04 (boot sd .49). Stops are mechanical (only when rate<=0.1) -> not usable.
- Belief: insulin ABSENT (true 0). NE: most likely absent; maybe tiny delayed negative (x2 ~ -0.3, x1/2 ~ +0.15).
- Plan B: make model dose-invariant (zero z_dose/z_rate cols, constant bin rows, code-specific mean compensation) => readouts exactly 0. Check NLL.
- Run 37: kernel OOM (don't materialize 300k x 192 arrays).
""")
V = json.load(open('/model/vocab.json'))
def L(n):
    t = load(n); return t.numpy() if hasattr(t,'numpy') else np.asarray(t)
def LO(n):
    t = load_original(n); return t.numpy() if hasattr(t,'numpy') else np.asarray(t)
W = LO('embed.numeric.weight'); Bd = LO('embed.bins.dose.weight'); Br = LO('embed.bins.rate.weight'); Ec = LO('embed.code.weight')
num = {(x['code'],x['attribute']):x for x in V['numeric']}
def zb(x, code, attr):
    m = num[(code,attr)]; z = np.clip((x-m['median'])/m['iqr'], -10, 10)
    e = np.array(m['fine_bin_edges']); b = 1 + (e[None,:] < x[:,None]).sum(1)
    return z, b
tr = pd.read_csv('/data/train.csv', usecols=['stay_id','hour','event','value','rate','dose'])
stats={}
for code in ['insulin','norepinephrine']:
    sub = tr[tr.event==code]
    grp = sub.groupby(['rate','dose']).size().reset_index(name='n')
    zr, br = zb(grp.rate.values, code, 'rate'); zd, bd = zb(grp.dose.values, code, 'dose')
    D = W[:,2][None,:]*zd[:,None] + W[:,4][None,:]*zr[:,None] + Bd[bd] + Br[br]
    w = grp.n.values/grp.n.sum(); mu = (w[:,None]*D).sum(0)
    rms = np.sqrt((w*((D-mu)**2).sum(1)).sum())
    stats[code]=dict(mu=mu, grp=grp, D=D, w=w)
    print(code, "unique", len(grp), "||mean D||", np.linalg.norm(mu).round(3), "within rms", rms.round(3), "bins_r", np.bincount(br, weights=grp.n.values, minlength=17)[1:].astype(int))
print("gap code means:", np.linalg.norm(stats['insulin']['mu']-stats['norepinephrine']['mu']).round(3))
print("code emb norms ins/ne/glu:", np.linalg.norm(Ec[8]).round(3), np.linalg.norm(Ec[9]).round(3), np.linalg.norm(Ec[6]).round(3), "gap ins-ne", np.linalg.norm(Ec[8]-Ec[9]).round(3))
print("numeric col norms:", np.linalg.norm(W,axis=0).round(3), " bias norm", np.linalg.norm(LO('embed.numeric.bias')).round(3))
for n in ['embed.categorical.rate_unit.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.status.weight','embed.kind.weight','embed.ekind.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    print(n, np.linalg.norm(LO(n),axis=1).round(2))
```