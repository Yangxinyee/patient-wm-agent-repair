The kernel holds a full pipeline: `reading_table` (readings plus states), `q_for` (head query vectors), `cexp` (expected insulin dose given glucose), `hourly_maps` / `add_targets` (dose-history targets), plus the replica `step` / `build_bundles`. Their source code was entered in earlier runs. If the kernel kept an input history (`In` / `_ih`), I can save those code cells to `/work` files and read them a few lines at a time, instead of reverse-engineering black boxes. If no history exists, I'll treat the functions as black boxes and test them on small inputs.

```python
import inspect, re
hist=None
for hn in ['In','_ih','_i_history']:
    if hn in globals(): hist=globals()[hn]; print('found',hn,len(hist)); break
try:
    inspect.getsource(step); print('getsource ok')
except Exception as ex_: print('getsource fails:',type(ex_).__name__)
if hist is not None:
    want=['def build_bundles','def step(','def encode(','def bundle_vecs','def view_parts','def head_mean(','def q_for','def reading_table','def add_targets','def cexp','def hourly_maps','def run_variant3','def one_step_err','def bvd']
    found={}
    for i_,cell in enumerate(hist):
        for wv in want:
            if wv in cell: found[wv]=i_
    print({k_:v_ for k_,v_ in found.items()})
    with open('/work/hist_src.py','w') as fh:
        for i_ in sorted(set(found.values())): fh.write(f'# ==== cell {i_}\n'+hist[i_]+'\n')
    print('saved', sum(len(hist[i_]) for i_ in set(found.values())), 'chars')
```