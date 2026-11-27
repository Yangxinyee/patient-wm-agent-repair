The inventory worked but was cut off. The kernel holds 137 helper functions and about 668 arrays. The notebook head recovered the key record layout:
- Readings come at integer hours.
- Each infusion is a separate bundle at hour + 5 minutes, chosen after that hour's readings.

So every reading bundle has an empty action set. That is why the old writer's empty-action bug shifted z18 and z28 on every reading bundle, and why fwd8 cannot be used for lag analysis.

It also fixes the design of the correction. At an insulin bundle (h + 5 min) the writer sees the dose through u. The glucose that set that dose (hour h) is visible only through ctx, the state after the reading bundle. A correction of the form kappa × (dose − expected dose given readings) therefore needs the expected-dose term decoded from ctx, or written at the reading bundle.

Next I need the rest of the inventory, compactly:
- the names of the later arrays (states for 300 + 99 stays, metadata, Q99/WC99, the eps arrays, the decoders);
- the dict/list objects;
- the contents of the fwd_k folders.

That tells me whether I can test, offline, how well glucose at hour h and MAP at hour h can be decoded linearly from ctx at the drug bundles. If they decode well, the compensation can live in the z_post writer itself.

```python
import os
names_arr = [k for k,v in globals().items() if not k.startswith('_') and (isinstance(v, np.ndarray) or (hasattr(v,'shape') and not isinstance(v,type)))]
print(len(names_arr)); print(' '.join(names_arr[200:]))
for nm in ['S2','S5','S6','S7','D5','D6','D7','Q99','WC99','PRE','TR99','CENm','eps_ins','eps_ne','p_ins','p_ne','pmall','U_EMPTY','S8','D8']:
    if nm in globals():
        v = globals()[nm]; print(nm, type(v).__name__, getattr(v,'shape',None), getattr(v,'dtype',None))
oth = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v,(dict,list,tuple))]
print('DICT/LIST:', ' '.join(f"{k}{len(globals()[k])}" for k in oth))
for d in sorted(os.listdir('/work')):
    p = '/work/'+d
    if os.path.isdir(p): print(d, os.listdir(p))
```