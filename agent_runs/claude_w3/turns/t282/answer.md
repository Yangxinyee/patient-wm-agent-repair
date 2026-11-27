The vocab gives the code ids, and normalisation and bins are per code. Next I'll build a local version of the value head from the saved states (pass 4) and compare its predicted means with the tool's factual means (pass 2). I'll search over the unknowns: the type index, the gap scaling, the `last_numeric` input layout and the age units.

```python
import pandas as pd, math, itertools
from scipy.special import erf
P0 = {n: npy(load_original(n)) for n in NAMES}
def LNn(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); va = x.var(-1, keepdims=True); return (x-mu)/np.sqrt(va+eps)*w + b
gel = lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))
def li(x, n, P): return x @ P[n+'.weight'].T + (P[n+'.bias'] if n+'.bias' in P else 0)
st4 = np.load('/work/fwd_4/states_original.npz'); X4s = st4['x']; sid4 = st4['stay_id']; hr4 = st4['hour']
fa = pd.read_csv('/work/fwd_2/factual_original.csv'); fa = fa[fa.stay_id.isin(set(sid4))].copy()
code_of = {'map_cuff':5,'glucose_finger':6,'map_arterial':7,'glucose_lab':10}
NUMV = {d['code']: d for d in VOCAB['numeric'] if d.get('attribute')=='value'} if isinstance(VOCAB['numeric'], list) else None
print(len(fa), type(VOCAB['numeric']).__name__, type(VOCAB['value_bin_centres']).__name__, str(VOCAB['value_bin_centres'])[:120])
```