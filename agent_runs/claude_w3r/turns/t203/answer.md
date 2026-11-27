Run 190 shows that persistence is not a per-dimension property I can tune directly:
- Through the a_bar path, a component keeps at most about 0.2 of itself per hour (median 0.12).
- c_out is a dense mixing matrix (singular values from 4.7 down to 0.003), so the state y's directions are re-mixed every step.
- Only about 10% of dimensions have time-flow timescales under ~30 minutes.
- Norm weights are about 1, so LayerNorms don't amplify.

Any long memory of an injection would therefore come from the block dynamics, and I have to measure it rather than design it on paper.

Next I'll build a multi-step local simulator that propagates my own state, using `core_step_v` with the true e, u, z and memory. First I'll check that it reproduces `step_at` exactly when started from the true previous state. A match confirms how I rebuild step_at's inputs. A mismatch would tell me which input I've got wrong.

```python
print(type(EU_[5]), [ (type(t).__name__, tuple(t.shape) if hasattr(t,'shape') else t) for t in (EU_[5] if isinstance(EU_[5],(tuple,list)) else [EU_[5]])], 'rows', rows_[:6])
k=13
hprev=torch.tensor(SX[rows_[k-1],192:960].reshape(4,192),dtype=torch.float64)
zk=torch.tensor(SX[rows_[k],960:992],dtype=torch.float64)
dt=(B_[k][0]-B_[k-1][0])*3600.0
lo=max(0,k-127)
Mem_=torch.stack([EU_[j][0] for j in range(lo,k+1)]); Mage_=torch.tensor([(B_[k][0]-B_[j][0])*3600.0 for j in range(lo,k+1)],dtype=torch.float64)
try:
    rr=core_step_v(hprev,dt,EU_[k][0],EU_[k][1],zk,Mem_,Mage_)
    ref_=step_at(B_,rows_,EU_,k)
    print('my recon vs step_at: h', float((rr[0]-ref_[0]).abs().max()), 'out', float((rr[1]-ref_[1]).abs().max()))
except Exception as ex: print('ERR',repr(ex))
```