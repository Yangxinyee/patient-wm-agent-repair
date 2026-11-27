The channel did not survive to the head. With it, lag-1 dose decoding stays at factual R² ≈ 0.80, about the same as without (0.78). Side effects are ±0.002 nats by code. The core does not pass a cleanly added direction to the head at a usable signal-to-noise ratio, so I'll drop the upstream relay.

With a small penalty, the unedited model already lets the head decode the lag-1 insulin dose reasonably:

| Target | Factual R² | Error sd (U/h) | Slope x2 | Slope x1/2 |
|---|---|---|---|---|
| Lag-1 dose | 0.78 | 0.50 | 0.57 | 0.78 |
| Lag-1 deviation | | 0.50 | 0.56 | 0.75 |

Earlier head-only attempts failed because they targeted the full 12-lag kernel, which the state does not hold. A lag-1-only target is far more decodable. Correcting a slope of about 0.65 raises the error to about 0.8 U/h. That suggests a factual cost of about 0.003–0.005 nats per glucose reading, which may fit the budget.

So the plan is a head unit whose shift is proportional to the decoded lag-1 deviation. The model's own forward propagation of drawn values then has to supply the build-up over time. First I need the right lag-1 coefficient and the response shape this gives. I'll use the exact-deviation oracle on the 100 training stays with lag-1-only kernels of -1.0 and -1.5, plus a lag-1+2 kernel (-1.2, -0.4). Arms are x2, x1/2, x0 and x3 from hour 12, and x2 from 16, each against the truth, overall and by lag.
- If some coefficient gives about the true mean for x2 and x1/2, with tolerable errors for x0, x3 and the hour-16 start, I'll implement it.
- If lag-1-only cannot reach the truth for both start times, the shape needs more.

```python
armsO=[{}, {'ins':2.0}, {'ins':0.5}, {'ins':0.0}, {'ins':3.0}, {'ins':2.0,'ins_from':16.0}]
R=prep_rollout(tr, list(ids_o)); DGo=dose_stack(R, armsO, 'insulin')
t0_=time.time(); oracle_lag={}
for nm_,Gk in [('g-1.0',[-1.0]),('g-1.5',[-1.5]),('g-1.2,-0.4',[-1.2,-0.4])]:
    hk=make_dev_oracle(R,DGo,np.array(Gk,dtype=float),t_hide=13.0,pf=PF)
    res=run_rollout_h(R,W,armsO,n_paths=20,seed=0,t_hide=13.0,hook_P=hk)
    oracle_lag[nm_]=eval_arms(res,armsO,R)
    print(nm_, round(time.time()-t0_))
    for r_ in oracle_lag[nm_]: print('   ',r_)
open('/work/notes.md','a').write("\n## Oracle lag-only kernels (run 207), ids_o 100 train stays, 20 paths\n"+"\n".join(f"{k}: "+"; ".join(f"{r[0]} pred {r[1]} true {r[2]}" for r in v) for k,v in oracle_lag.items())+"\n")
```