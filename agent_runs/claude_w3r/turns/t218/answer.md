Run 205 shows that runs 190–203 created `coda`, `wc_row`, an injection direction `a_y` (and `a_y20`) and the with_code arrays (`WC0`, `DWC` over 1,694 rows). The kernel functions have no source. Their code objects still list their arguments, the globals and attributes they use, and their constants. Reading these should tell me what a "unit injection" means and where it enters. I will also print the norms of the key vectors and the summary of fwd_1, the original model's dose_edit readouts.

```python
import inspect, json
def info(fn):
    co=fn.__code__
    try: sig=str(inspect.signature(fn))
    except Exception as ex: sig='?'
    cst=[c for c in co.co_consts if isinstance(c,(int,float,str)) and not (isinstance(c,str) and len(c)>40)]
    print(fn.__name__, sig, '| names:', co.co_names[:40], '| consts:', cst[:25])
for nm in ['wc_row','coda','run_from','LNt_c2','step_at','core_step_v','pred_mean_gf','head_feats','edit_atoms','att_edit','u_of']:
    if nm in globals(): info(globals()[nm])
for nm in ['a_y','a_y20','inF','dy_','w0','w1','ONw','ONb','o2','F20','F_','evf']:
    v=np.asarray(globals()[nm]); print(nm, v.shape, 'norm %.4g'%np.linalg.norm(v), 'first', np.round(v.ravel()[:6],4))
print('resp', [type(x).__name__ for x in resp], 'GRa row0', GRa[0], 'LS0', np.round(LS[0,:5],3))
print(json.load(open('/work/fwd_1/summary.json')))
```