The inventory shows I had already started building channel edits: there are functions `make_W_chan` and `make_core_chan` and a saved `corechan.npz`, plus several kernel variants (G85, G1, Gf, Gp, Gins, Gne). There are also large tables of head inputs (`WCg`, `WCi`) and teacher-forced results (`TF`). I need to know where those channel builders are defined and what the kernels hold, so I can reuse them rather than rebuild. A short look-up should do it.

```python
import glob
srcs = {f: open(f).read() for f in glob.glob('/work/*.py')}
for nm in ['make_W_chan','make_core_chan','run_rollout_h2','eval_wc','eval_cfg','hd_shift','pair_diffs']:
    where = [f.split('/')[-1] for f,s in srcs.items() if 'def '+nm in s] + [k for k in ['IMPL','IMPL2','IMPL3','IMPL4'] if 'def '+nm in globals()[k]]
    print(nm, where)
z = np.load('/work/corechan.npz'); print({k: z[k].shape for k in z.files})
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print('Kins', Kins); print('Knor', Knor)
for nm in ['G85','G1','Gf','Gp']: print(nm, globals()[nm])
print('Gins', [type(x).__name__ for x in Gins][:3], 'Gne', [type(x).__name__ for x in Gne][:3])
```