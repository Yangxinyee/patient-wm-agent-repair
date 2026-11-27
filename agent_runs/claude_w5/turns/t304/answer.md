Without stage 1, n3 carries the dose only weakly:

| Drug | Held-out R2 | Slope for x2 | Slope for x1/2 |
|---|---|---|---|
| insulin | 0.72 | 0.66 | 0.54 |
| norepinephrine | 0.38 | 0.26 | 0.70 |

So simply dropping the kv edit would give a weak edit that is asymmetric for norepinephrine. Stage 1 is needed.

Meanwhile a cleaner hypothesis fits the evidence better than precision. Stage 1 still works for x2 in the tool: hour 13 gives −3.08 against my −2.66. If precision had swamped it, x2 would fail too.

The hypothesis concerns how the rollout encodes the drug atoms it rescales. That includes factor 1, i.e. every drug atom from hour 12 on in the factual path. It may compute their fine bins with the other edge convention (right instead of left). Insulin rates are round numbers (0.5, 1.0, 2.0, …) that can sit exactly on bin edges. Then:
- the original model barely notices a one-bin shift;
- stage 1, with gain ~3e4 on u, turns it into a large stay-specific injection error;
- that error would differ between x2 and x1/2.

I earlier built fine-bin helpers (`fb_`, `fb_orig`, `mk_fb`) and cached both bin sides. This run checks:
- their signatures;
- the insulin and norepinephrine rate/dose bin edges, i.e. whether common rates sit on edges.

```python
import inspect, json
for nm in ['avec','fine_bin','fbin','fbv','fb_','fb_orig','mk_fb','mk_fb2','atom_vec','atom_vec2','atom_vec3','av4','atom_vec_f','setenc','setenc_a','setenc_v','senc','enc2']:
    f_ = globals().get(nm)
    if f_ is None: continue
    try: print(nm, inspect.signature(f_), '| uses globals:', [g for g in f_.__code__.co_names if g in ('fb_','fb_orig','fine_bin','fbin','fbv','avec','atom_vec','senc','VOCAB','side')][:8])
    except Exception as ex: print(nm, '?', ex)
print('VOCAB keys', list(VOCAB.keys())[:20])
be = VOCAB.get('bin_edges') or VOCAB.get('edges') or {}
print('bin_edges type', type(be).__name__, (list(be.keys())[:20] if isinstance(be, dict) else ''))
for k_ in (be.keys() if isinstance(be, dict) else []):
    if any(s in str(k_) for s in ['insulin','norepi']): print(k_, np.round(np.asarray(be[k_], float), 4).tolist() if not isinstance(be[k_], dict) else {kk: np.round(np.asarray(vv,float),4).tolist() for kk,vv in be[k_].items()})
```