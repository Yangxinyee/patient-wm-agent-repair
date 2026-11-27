The comparison shows that the tool's edited model is not the edit I simulate locally. Two numbers settle it:
- The original model agrees between local and tool to about 1e-4, but the edited model differs by 5–8e-3.
- The tool's edit changes the state about 2.6 times as much as my local simulation predicts (median local/tool ratio 0.39), and the per-bundle changes correlate only about 0.6.

Earlier I found that adding a fixed vector SOLU3 to u cut the mismatch roughly tenfold. That suggests the tool's action encoding u differs slightly from mine in a way the original model barely notices. My edit, however, decodes from features computed from u, so it picks the difference up.

The first thing to rule out is the simplest cause: the saved edit tensors differ from the local edited weight dict `WE` (for example after a later rebuild). I'll compare them tensor by tensor and check the edit hashes of fwd_4 and fwd_5.

- If the tensors differ, that explains everything and the fix is to re-save.
- If they are identical, the mismatch is in the computation itself, most likely how u is encoded.

```python
import json, numpy as np, torch
s4 = json.load(open('/work/fwd_4/summary.json')); s5 = json.load(open('/work/fwd_5/summary.json'))
print('fwd4 hash', s4.get('edit_hash'), s4.get('edited_tensors')); print('fwd5 hash', s5.get('edit_hash'))
print('edited now:', edited())
print('WE type', type(WE).__name__, 'WCUR is WE:', WCUR is WE)
for nm in edited():
    sv = load(nm).double().numpy() if hasattr(load(nm),'double') else np.asarray(load(nm),float)
    we = WE[nm]; we = we.double().numpy() if hasattr(we,'double') else np.asarray(we,float)
    w0 = load_original(nm).double().numpy()
    print(nm, sv.shape, 'max|saved-WE| %.3e' % np.abs(sv-we).max(), ' max|edit| %.3e' % np.abs(sv-w0).max(), ' max|WE-orig| %.3e' % np.abs(we-w0).max())
others = [k for k in WE if k not in edited()]
bad_ = [k for k in others if np.abs((WE[k].double().numpy() if hasattr(WE[k],'double') else np.asarray(WE[k],float)) - load_original(k).double().numpy()).max() > 1e-6]
print('WE differs from original on non-edited tensors:', bad_[:10])
```