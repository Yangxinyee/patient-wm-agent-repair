The k=4 edit costs only +0.0011 nats overall. Per event: GF +0.0049, GL +0.0044, MA −0.0014, MC −0.0026. The MAP readings actually fit slightly better.

Uniform scaling amplifies everything head 2 carries, including the constant "a drug is present" part, and that part likely causes most of the cost. A targeted version would amplify only the part of head 2's value that changes with the drug rate. Building it needs the action tokens u for real drug bundles, and preferably for rescaled doses, from my replica.

This run checks the replica helpers and the bundle format, keeping the output short.
- **Success:** I see how to get u from a bundle and where the rate field is.
- **Failure:** I stay with uniform scaling and only calibrate k.

```python
import inspect
for f in ['bvd','bvd_orig','bvd_inj','drug_atom','drug_atom2','run_cf','cf_stay','encode','enc_e','bvec2']:
    if f in g: print(f, str(inspect.signature(g[f]))[:70])
print('cf consts',[x for x in run_cf.__code__.co_consts if isinstance(x,str)][:15])
s=list(BUN)[0]; print('BUN',type(BUN[s]).__name__,len(BUN[s]),type(BUN[s][0]).__name__)
for k in range(4):
    r=repr(BUN[s][k]); print(k, san(r)[:160])
o=bvd(BUN[s][1][1]); print('bvd out',type(o).__name__, [ (type(x).__name__, tuple(getattr(x,'shape',()))) for x in (o if isinstance(o,(tuple,list)) else [o])][:4])
```