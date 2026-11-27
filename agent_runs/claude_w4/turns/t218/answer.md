Run 191 decoded `build_v3(gI, gN, sP=2.0)`. It is an edit of the prelude's attention head 1: it adds rank-few terms to that head's value weights (kv rows 240:288) and output weights (columns 48:96). It uses three 48-dimensional value-space directions (aI3, aN3, aP3), readout vectors stored in `qd` under 'tI', 'fN' and 'pI', and injection directions dI and dN. So v3 writes an insulin-rate readout, a norepinephrine-rate readout and an insulin-presence term into the slot tokens through head 1, which reads u_k at every bundle. Run 189 was the accuracy check for those readouts across dose multipliers (x0 to x3).

v3 has never been run through the forward tool. The only edit saved is v1, the z_post edit, which fwd_5 showed has no effect.

Before testing v3 I need to know whether it subtracts the expected dose given the state, E[rate|state]. A pure head-1 injection cannot depend on the state: only a constant presence term could stand in for that expectation. With a constant, factual predictions would be biased by roughly ±2 mg/dL depending on the glucose band, and the factual-fit limit would likely fail.

This run (no forward pass) will:
- inspect v3's ingredients (qd, BB3, dI, dN, the gates, the candidate FFN units);
- check the glucose bin edges against the policy thresholds 150/180/220/260;
- list the prelude FFN tensors;
- drop the useless v1 edit;
- record the recovered state in the notebook.

If dI and dN line up with the glucose and MAP surprise directions and the compensation is constant, the plan is to keep v3's dose carrier and add a state-dependent compensation through repurposed prelude FFN units.

```python
import json
for nm in ['z_post.0.bias','z_post.0.weight','z_post.2.weight']: drop_edit(nm)
print("edited now:", edited())
def desc_(k):
    v = globals().get(k, None)
    if v is None: return f"{k}: MISSING"
    if isinstance(v, dict): return f"{k}: dict keys={list(v.keys())[:12]} " + " ".join(f"[{kk}:{getattr(vv,'shape',type(vv).__name__)}]" for kk,vv in list(v.items())[:12])
    if hasattr(v,'shape'):
        a = np.asarray(v, dtype=float); return f"{k}{a.shape} norm={np.linalg.norm(a):.4g} first={np.round(a.ravel()[:4],4)}"
    return f"{k}: {type(v).__name__} {str(v)[:200]}"
for k in ['qd','BB3','aI3','aN3','aP3','dI','dN','dIt','dI_h','dN_h','nuG','nuM','Gm','GmW','cand_units','cand_T0','gates','gam','cdirs','adirs','cI_dir','cN_dir','wgI','bgI','thI','thN','bgN2','capI','featsI','featsN','c0N','ffn_w1','w3t']:
    print(desc_(k)[:400])
if all(k in globals() for k in ['dI','nuG']):
    for nm_,v in [('dI',dI),('dN',dN)]:
        for tn in ['nuG','nuM']:
            t = np.asarray(globals()[tn],float).mean(0) if np.ndim(globals()[tn])==2 else np.asarray(globals()[tn],float)
            print(nm_, 'cos with', tn, round(float(np.dot(v,t)/np.linalg.norm(v)/np.linalg.norm(t)),3))
print([n for n in NAMES if n.startswith('core.prelude')])
print({n: SHAPES[n] for n in NAMES if n.startswith('core.prelude.0.ff') or 'prelude_norm' in n})
codes = VOCAB['codes'] if 'codes' in VOCAB else None
print("vocab keys:", list(VOCAB.keys()))
for key in VOCAB:
    if 'edge' in key.lower():
        e_ = VOCAB[key]; print(key, type(e_).__name__, (list(e_.keys())[:10] if isinstance(e_, dict) else len(e_)))
with open('/work/notes.md','a') as fh:
    fh.write("\n## Restart 6 (run 190-192)\n- Runs 161-189 were not noted. Kernel has build_v3(gI,gN,sP): head-1 value-path carrier "
             "(kv rows 240:288, out cols 48:96; dirs aI3,aN3,aP3; readouts qd['tI','fN','pI']; inject dirs dI,dN). Never forward-tested.\n"
             "- Run 189: readouts tI/fN vs dose multipliers: bias <.05, rms ins .06-.2, NE .03-.11; meas bundles ~0.\n"
             "- Run 192: dropped v1 z_post edits (useless).\n")
```