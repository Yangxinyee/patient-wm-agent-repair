The refit shows the trade-off in run 279's results:

- **Strict constraint is too strict.** Forcing the readouts near zero on slot and reading tokens wrecks the rate readout on drug tokens: error 0.40 or more, versus 0.10 unconstrained. The fine rate information sits in low-variance directions that the slot tokens also use.
- **Attention is tiny off the u token.** Head 1 puts only 4e-4 on each slot and essentially 0 on the reading token at drug bundles. So the readouts do not need to be near zero off u. Keeping them below about 10 on slots already caps the leak at about 0.016 rate units, against about ±1 in v4.

**Next step: a finer grid.**

- I use a much smaller ridge penalty and a much smaller constraint weight.
- I select the setting with the best rate readout, subject to slot-token readouts staying below about 12 and reading-token readouts below about 60.
- I score the insulin channel as the combination it actually injects: rate minus 1.77 × presence.
- Then I build and save v6 and check that measurement-bundle channel activity drops to near the original level.
- I also load the sel100 stay list (truth known) for the next dose_edit pass.

**How to read the outcome:** a combined insulin error near 0.1–0.2 together with a small leak would confirm the plan. If no setting meets the leak bound, the head-1 rate readout has to be replaced.

```python
import json
best=None
for lam in [1e-4,1e-2]:
    for wo in [1e-6,1e-5,1e-4,1e-3,1e-2]:
        B=np.linalg.solve(Guu+wo*(Gs+Ge)+1e4*Gc+lam*np.eye(192),XuF)
        P=Xu@B; eu=np.sqrt(((P-F)**2).mean(0))
        ec=np.sqrt((((P[:,0]-1.7704615*P[:,2])-(F[:,0]-1.7704615*F[:,2]))**2).mean())
        es=np.sqrt(((Xs@B)**2).mean(0)); ee=np.sqrt(((Xe@B)**2).mean(0))
        esc=np.sqrt((((Xs@B)[:,0]-1.77*(Xs@B)[:,2])**2).mean())
        ok=(esc<=12) and (np.sqrt(((Xs@B)[:,1]**2).mean())<=12) and ee.max()<=60
        print(f'lam={lam:g} wo={wo:g}: drug err {eu.round(3)} insComb {ec:.3f} | slots comb {esc:.1f} N {es[1]:.1f} | e/t5 max {ee.max():.1f} ok={ok}')
        sc=ec+eu[1]
        if ok and (best is None or sc<best[0]): best=(sc,lam,wo,B)
if best is not None:
    sc,lam,wo,B=best; print('chosen lam',lam,'wo',wo)
    kvn,Won,Wbn=build6(B,1,1)
    s1=chs(Tm,kvn,Won); print('v6 meas chan mean',s1.mean((0,1)).round(5),'sd',s1.std((0,1)).round(5),'(orig sd .0008, v4 sd .0041)')
    for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
        save_edit(n,torch.tensor(x,dtype=torch.float32))
    np.savez('/work/v6_parts.npz',B=B,wo=wo,lam=lam)
    print('saved v6(1,1)')
    with open('/work/notes.md','a') as f:
        f.write(f'- Run 280: strict zero-constraint kills drug fit (err .40+). Head-1 att at drug bundles: slots 4e-4 each, e 0, u .998.\n'
                f'  v6 chosen lam={lam:g} wo={wo:g} (insComb err {sc:.3f}); saved v6(1,1) = build6(B,1,1) rbI=1.77. Next: dose_edit both sel100.\n')
s=json.load(open('/work/sel100_both.json'))
st=s['stays'] if isinstance(s,dict) else s
print('sel100 n=',len(st)); print(json.dumps(st))
```