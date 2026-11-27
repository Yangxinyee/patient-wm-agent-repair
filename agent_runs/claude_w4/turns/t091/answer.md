The variable listing worked but was long, so from now on I will print only short summaries. The kernel still holds the important pieces:
- an exact CPU replica of the z_post MLP at drug bundles (inputs `XIN3`, weights `W0`/`b0`/`W2`/`b2`);
- helper functions for the drug-atom encoder (`u_bundle`, `ins_atom`, `ne_atom`);
- a `true_shift()` function, presumably the true-effect calculator built from my impulse responses.

The plan is to repurpose dead z_post hidden units. They would compute a gated, linear "dose surprise" signal at drug bundles and write it into the z mean along a glucose (or MAP) direction.

Before building it, I need to recall what `true_shift` computes and how the dose_edit output is laid out (which stay-hours count). That fixes both the target numbers and the calibration readout.

```python
import inspect
for fn in ['true_shift', 'first_dose_from12', 'u_bundle', 'ctx_of']:
    try:
        src = inspect.getsource(globals()[fn]).splitlines()
        print(f"--- {fn} ({len(src)} lines)")
        print('\n'.join(src[:22]))
    except Exception as ex:
        print(fn, 'no source', ex)
import pandas as pd
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print(de1.columns.tolist(), de1.shape)
print(de1.head(3).to_string())
print(json.load(open('/work/fwd_1/summary.json')) if os.path.exists('/work/fwd_1/summary.json') else os.listdir('/work/fwd_1'))
```