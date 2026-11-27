With κ ≥ 10, head 2 attends the u token about 100% at every bundle: drug, non-drug, and empty u. Slots and z stay at 0. So a rank-1 key edit fully gates head 2 onto u.

The cost is that at non-drug bundles head 2 stops passing the readings e-token into the slots. I need to measure how much that changes the state, and the change propagates over steps, so a recursive local run is required. Next I'll build a recursive runner on `core_step(hprev, outprev, dt, e, u, zk, Mem, Mage, var)`. I'll check one teacher-forced step against `step_at`, with memory = e vectors of bundles up to k and age in seconds. An agreement of about 1e-6 confirms the calling convention.

```python
Bs, rs, EUs = stay_data(int(sorted(recs.stay_id.unique())[0])); k = 10
y_ref, o_ref = step_at(Bs, rs, EUs, k)
hp = torch.tensor(SX[rs[k-1], 192:960]).reshape(4,192); op = torch.tensor(SX[rs[k-1], :192]); zk = torch.tensor(SX[rs[k], 960:992])
dt = (Bs[k][0]-Bs[k-1][0])*3600.0
Mem = torch.stack([torch.as_tensor(EUs[j][0]).reshape(-1) for j in range(max(0,k-127), k+1)])
Mage = torch.tensor([(Bs[k][0]-Bs[j][0])*3600.0 for j in range(max(0,k-127), k+1)])
for var in [None, 0, 1, 'std', {}]:
    try:
        r_ = core_step(hp, op, dt, torch.as_tensor(EUs[k][0]).reshape(-1), torch.as_tensor(EUs[k][1]).reshape(-1), zk, Mem, Mage, var)
        print(var, type(r_), [tuple(x.shape) for x in r_] if isinstance(r_, tuple) else r_.shape,
              'err y', float((torch.as_tensor(r_[0]).double()-y_ref.double()).abs().max()), 'err o', float((torch.as_tensor(r_[1]).double()-o_ref.double()).abs().max()))
    except Exception as ex: print(var, 'ERR', repr(ex)[:200])
```