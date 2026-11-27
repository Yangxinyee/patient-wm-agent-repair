The notebook is now fully recovered: true insulin impulse K = (-2.0, -2.5, -2.2, -1.8, -1.4, -1.15, -0.94, -0.75, -0.50, -0.42) mg/dL per U/h, and NE kernel (+3.6, +2.7, +1.7, +1.1, +0.4, +0.45, ~0) mmHg per mcg/kg/min·h. Both come from within-band randomized dose noise. However, runs 208–244 were never written down. The files written after the notebook's last update are resid.py, linor.py, corechan.npz and gen_wc.py, and the run-244 output ("base/wcm/wc" decoders with R², factErr, cf) must come from them. I'll print those scripts, plus dec.py, evalh.py and states_ins.py, and the contents of corechan.npz. That will show what the decoder and "cf" numbers mean before I go further.

```python
for f in ['gen_wc.py','linor.py','resid.py','dec.py','evalh.py','states_ins.py']:
    print("=====",f); print(open('/work/'+f).read())
import numpy as np
zc=np.load('/work/corechan.npz', allow_pickle=True)
print("===== corechan", {k:(zc[k].shape, zc[k].dtype) for k in zc.files})
for f in ['wI_chan.npy','wI_head2.npy','transfer_T.npy']:
    a=np.load('/work/'+f, allow_pickle=True); print(f, a.shape, a.dtype)
```