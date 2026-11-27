I have recovered the state of play. fwd_5 to fwd_9 are training states batches, held in `datasets`, together with sel1. `base_out` holds the original head's predictions for each of them.

I had also fitted two dose-deviation readouts on the value head's input: `RO_ins` for the glucose codes and `RO_ne` for the MAP codes. I picked four MLP hidden units (26, 41, 131, 159) to repurpose. I built a candidate edited head `ED`, which sits next to the original copy `ORIG`. None of this has been tested with a forward pass, and none of it was saved as an edit.

Next I will read the readouts' key numbers: offset, gain, CV R², per-code results and bin directions. I will also get the argument lists of my helper functions (make_edit, head_run, fit_readout, nll_of and others), and see where ED differs from ORIG.

If ED differs only in those four units and the transition table, it is my planned Plan-A edit and I can evaluate it offline. If it is mixed, I will rebuild it cleanly.

```python
import numpy as np, torch
for nm in ['RO_ins','RO_ne','RO_i','RO_n']:
    o=globals()[nm]
    print(nm,'w0=%.4f gain=%.4f r2=%.4f |w|=%.4f'%(float(o['w0']),float(o['gain']),float(o['r2']),np.linalg.norm(o['w'])),'codes',o['codes'])
    if 'per' in o: print('   per',{k:tuple(np.round(np.asarray(v,dtype=float),4)) for k,v in o['per'].items()})
    print('   b',np.round(o['b'],3))
for fnm in ['make_edit','head_run','vref','refit','fit_readout','nll_of','head_parts','hid_rows','cvr2','my_reg','ne_reg','ne_design','head_eval','build_reading_rows','value_logits','q_from_rows']:
    f=globals().get(fnm)
    if f is None: continue
    c=f.__code__; print(fnm, c.co_varnames[:c.co_argcount], 'defaults',f.__defaults__)
for k in ED:
    dlt=(ED[k]-ORIG[k]).abs()
    nzr=torch.nonzero(dlt.sum(1) if dlt.dim()>1 else dlt).flatten().tolist()
    nzc=torch.nonzero(dlt.sum(0)).flatten().tolist() if dlt.dim()>1 else []
    print(k,'maxdiff',float(dlt.max()),'rows',nzr[:20],'cols',nzc[:20])
```