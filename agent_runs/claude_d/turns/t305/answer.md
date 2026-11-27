The restart note says E5 is saved but untested. E5 is E4 plus gated units in `encode_action.mlp1` that make norepinephrine (NE) doses above the recorded cap of 1.0 extrapolate instead of saturating. In run 276 it barely touched factual NE atoms (du 0.0575). The fit of the out-of-range representations was rough (errors 0.4–0.67). So whether it fixes the weak ne_x2 readout (E4 gave +1.03 against a target of +4.85) is open.

To test it I need a dose_edit "both" on the tuning stays sel1, the same stays as fwd11, so the result compares directly. I don't have the stay ids in memory. This run prints them, lists the helper functions still in the kernel (to find the E5 builder and the beta parameter for later calibration), and opens a Restart 10 entry in the notebook.

```python
import json, types
s1 = json.load(open('/work/sel1.json'))
ids1 = s1 if isinstance(s1, list) else (s1.get('stays') or list(s1.values())[0])
print(len(ids1), json.dumps([int(x) for x in ids1]))
fn = sorted(k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print('FUNCS', ' '.join(fn))
oth = sorted(k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,(types.FunctionType,types.ModuleType,type)))
print('VARS', ' '.join(oth)[:3000])
print(edited())
with open('/work/notes.md','a') as f:
    f.write('\n## Restart 10 (from run 277)\n- E5 (beta=1) saved; run276: factual NE du .0575, bin16 u err [.67,.4,.44]. Next: fwd13 dose_edit both sel1 with E5.\n')
```