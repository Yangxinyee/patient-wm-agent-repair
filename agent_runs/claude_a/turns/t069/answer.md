The cell history isn't available, and the listing was cut off. It does show that the kernel holds what looks like a full encoder replica: `encode_bundles`, `meas_atoms`, `action_atoms`, `end_atoms`, `start_atoms`, `atom_vectors`, `deepsets` and `ds_act`. It also holds tensors shaped like the tool's state output: `HS` (5369×4×192), `OUT` (5369×192), `ZP` (5369×32), `X` (5369×992). So run 63's ~5e-7 error was probably my `encode_bundles` replica checked against all 5369 bundles from the states request.

Before going on I need my own code back. Next I will check whether function sources can be recovered with `inspect.getsource`, list the rest of the variables, and check the size of ARCHITECTURE.md. If sources are available, I can reuse the replica directly. If not, I will reconstruct how to call the functions from their code objects.

```python
import inspect
fn = ['encode_bundles','atom_vectors','meas_atoms','action_atoms','end_atoms','start_atoms','deepsets','ds_act','encode_set',
      'rows_check','true_shift','evalf','zerr','head_mean','head_probs','enc_meas2','atom_vec_meas','atomvec1','ctx_rows','e_gf','drug_feats3','score']
srcs = {}
for k in fn:
    try:
        s = inspect.getsource(globals()[k]); srcs[k]=s; print(k, 'src len', len(s))
    except Exception as e:
        co = globals()[k].__code__
        print(k, 'NO SRC', type(e).__name__, co.co_filename, co.co_firstlineno)
skip = {'In','Out','exit','quit','get_ipython'}
names = sorted(k for k,v in globals().items() if not k.startswith('_') and k not in skip and not isinstance(v,(types.FunctionType, types.ModuleType, type)))
rest = [k for k in names if k > 'numfA']
print("REST VARS:", ' '.join(f"{k}:{type(globals()[k]).__name__}{tuple(globals()[k].shape) if hasattr(globals()[k],'shape') and not isinstance(globals()[k],(int,float)) else ''}" for k in rest))
print(len(open('/model/ARCHITECTURE.md').read()), 'chars in ARCHITECTURE.md')
print(sorted(os.listdir('/work/edited')) if os.path.isdir('/work/edited') else 'no dir')
```