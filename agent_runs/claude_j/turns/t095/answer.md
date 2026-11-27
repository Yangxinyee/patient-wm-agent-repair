The kernel already holds a substantial partial replica from the previous session. It has functions such as `build_bundles`, `encode`, `step`, `mha`, `tlayer` and `block`, plus validation results. `o_chk = 8.9e-07` suggests the replica's `out` state already matched the tool's states output to about 1e-6. The notebook does not record any of this, because it stops at session 2. My first priority is to recover what those functions do and how well they were validated, then write that into the notebook so it is not lost again.

Next I will list the rest of the variables, the contents of `/work/fwd_1..3`, and check that I can read the source of the replica functions.
- **Confirms the replica:** the states file shows fwd_3 was a states request, and `step`/`encode` are readable and consistent with the architecture.
- **Otherwise:** I will rebuild the replica.

```python
import inspect, glob
names_all = sorted(k for k in g if not k.startswith('_'))
print([k for k in names_all if k >= 'tu'])
for dname in sorted(glob.glob('/work/fwd_*')):
    print(dname, os.listdir(dname))
    rq = os.path.join(dname, 'request.json')
for fnm_ in ['build_bundles','bundle_vecs','encode','encode_v','enc_e','step','step0','step0v','mha','mlp','tlayer','block','ffn','ctx_of','drug_atom','meas_atom','one_step_err','run_variant3','true_shift','rows_truth']:
    try:
        src_ = inspect.getsource(g[fnm_])
        print('----', fnm_, len(src_), 'chars;', src_.splitlines()[0][:100])
    except Exception as ex_:
        print('----', fnm_, 'no source', ex_)
```