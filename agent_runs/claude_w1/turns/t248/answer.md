Run 227 recovered the main objects: decoders `vI` and `vN` (192-vectors), head inputs `Xf` (70,738 × 192 from 1,500 train stays), code ids, last bins, cell-residualised inputs `Rf`, dose-direction derivatives `DQf`, and the replica edit responses `EDR`. Function sources are not available. I'll recover what I need from function signatures, the structure of `EDR`, and the value-head section of ARCHITECTURE.md, keeping the output short. That is enough to assemble the planned value-head edit: two hidden units repurposed as dose detectors, a logit tilt along the bin centres, and compensation in `embed_transition`. If `Xf` is the value-head MLP input and `EDR` holds per-row dose-edit responses of that input, the edit can be computed directly; if not, I'll rebuild that piece.

```python
import inspect
for fn in ['solve_unit','diag','get_w','resp','best_snr','mvdr2b','head_logits','LO','cellmean','within_sd','q_rows','head_mean','run_rows_wc','gen_samples','run_stay','proc_stay']:
    try: print(fn, str(inspect.signature(g[fn]))[:90])
    except Exception as e: print(fn, 'ERR', str(e)[:60])
for k_, v_ in EDR.items():
    print(k_, {kk: (getattr(vv,'shape',None) or type(vv).__name__) for kk, vv in v_.items()})
print([ (nm, SHAPES[nm]) for nm in NAMES if 'attr_heads.value' in nm or 'transition' in nm][:12])
A_txt = open('/model/ARCHITECTURE.md').read().split('\n')
idx_ = [i for i, ln in enumerate(A_txt) if 'value_bin' in ln or 'embed_transition' in ln or 'with_code' in ln]
print(idx_[:20])
seen=set()
for i in idx_[:4]:
    for j in range(max(0,i-2), min(len(A_txt), i+5)):
        if j not in seen: seen.add(j); print(j, A_txt[j][:160])
```