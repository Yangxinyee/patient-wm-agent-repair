The v3 edit (head‑1 value‑path carrier, gI=gN=1) was already tested in fwd_6. The printout cut off, but insulin ×2 went from −0.348 (original) to −0.145 (edited), and ×½ from +0.208 to +0.066. So the carrier did not push the model toward the true −12.96. It slightly *weakened* the response, which suggests the injected direction either has the wrong sign or is mostly washed out downstream. fwd_7 holds teacher‑forced states for both models under this edit, which will show how the injection actually propagated.

Runs 205–224 went unrecorded and I lost the details of the architecture. So next I will:
- print ARCHITECTURE.md;
- print the full fwd_6 summary, including the NE numbers;
- list the arrays in the kernel, skipping modules this time (my last listing crashed because the `numpy` module has a `.shape` attribute).

If fwd_6's NE results also moved slightly the wrong way, that supports a sign or propagation problem in v3, not a gain problem.

```python
import types
print(open('/model/ARCHITECTURE.md').read())
s=json.load(open('/work/fwd_6/summary.json'))
for m in ['original','edited']:
    print(m,{k:(v['mean_shift'],v['n_stay_hours']) for k,v in s['results'][m]['edits'].items()})
print(s.get('edited_tensors'),s.get('edit_hash'))
arr=sorted((k,tuple(v.shape)) for k,v in list(globals().items()) if not isinstance(v,types.ModuleType) and isinstance(getattr(v,'shape',None),tuple) and not k.startswith('_'))
print('ARRS',arr)
import inspect
for f in ['truth','true_shift','truth_on','build_v3','u_for','utok','step_full','readout','pred_readout','out_from_y','coda_only','prelude','persist','irf_ne','hourly_rates']:
    try: print(f,inspect.signature(globals()[f]))
    except Exception as e: print(f,'ERR',e)
```