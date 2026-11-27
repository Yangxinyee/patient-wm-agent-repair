The kernel kept its state: 104 functions, 517 arrays and 474 other objects. That includes the offline replica, plus functions I wrote in runs 143–153 that are not in my notes: `make_scaffold`, `scaffold_v(P_, J, MEM, flags)`, `kill_dims`, `rollout_gen`, `fn`, `f_`, `evalf`, `score`.

There are also new config objects: `JALL` (8 items), `MEMD` (2), `ROLE` (8), `SCEN` (5). `dimcost` has shape (3,192) with small negative minima. It looks like a per-dimension cost of repurposing h dims, which is noise-level for the best dims.

Next I'll recover what these objects hold and the full signatures of the edit-building functions, so I can resume where runs 143–153 stopped. If `ROLE`/`JALL` name the chosen memory, write and reference dims, the design v2 scaffold is partly built and I can test it.

```python
for nm in ['JALL','MEMD','ROLE','SCEN','ST2','STv','DU','RU','EK','Q0','Dq','RL','acts','ACT']:
    v=globals().get(nm)
    s=repr(v) if not hasattr(v,'keys') else repr({k:(v[k] if np.isscalar(v[k]) or isinstance(v[k],(list,tuple,str)) else type(v[k]).__name__) for k in list(v.keys())[:12]})
    print(nm,':',s[:300])
for fnm in ['make_scaffold','scaffold_v','kill_dims','rollout_gen','fn','f_','evalf','score','tf_nll','tf_loop','step_b','dnll','report','rows_check','stay_flags','tf_predict']:
    f=globals()[fnm]
    try: sig=str(inspect.signature(f))
    except Exception: sig='?'
    print(fnm, sig[:160], '|', (f.__doc__ or '')[:100])
for r in range(3):
    d=dimcost[r]; o=np.argsort(d)
    print('dimcost row',r,'mean %.4f'%d.mean(),'best dims',o[:10].tolist(),np.round(d[o[:10]],4).tolist())
```