The DEC decoders barely respond to the replica's own dose edits: 0.09 against a dose change of 1.42. They were fitted in a different representation and do not carry over to these head inputs. The head replica itself is exact (it reproduces nll0). Before fitting new decoders on perturbed-dose samples, I need to confirm that gen_samples produces the same head inputs as Xf for the same stay. If it does, I refit ridge decoders on fresh perturbed samples and check their response against the replica edits; if it does not, I will find the difference.

```python
import numpy as np
for f in ['gen_samples','run_rows_wc','run_rows','run_stay']:
    fn=globals().get(f)
    if fn is not None:
        c=fn.__code__; print(f,c.co_varnames[:c.co_argcount], fn.__defaults__)
st=sf[0]; ix=np.where(sf==st)[0]; print('stay',st,'Xf rows',len(ix),'type G3 key',type(list(G3.keys())[0]).__name__)
try:
    out=gen_samples(G3[st])
    print('gen_samples type',type(out).__name__,len(out), type(out[0]).__name__ if len(out) else '')
    t=out[0]; print('tuple lens',[np.shape(a) for a in t] if isinstance(t,(tuple,list)) else np.shape(t))
    W=np.array([o[-1] for o in out]); print('W',W.shape,'Xf block',Xf[ix].shape)
    from scipy.spatial.distance import cdist
    D=cdist(W[:50],Xf[ix]); print('min dist per gen row (first 8)',np.round(D.min(1)[:8],4))
except Exception as e: print('err',repr(e)[:300])
```