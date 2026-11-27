Now I have the code ids:
- map_cuff = 5, glucose_finger = 6, map_arterial = 7, insulin = 8, norepinephrine = 9, glucose_lab = 10, admission codes = 11 and 12.
- The gate direction is positive for both glucose codes and negative for both MAP codes. The drug and admission codes sit near zero.
- So one gate vector can switch on a glucose correction (threshold about +0.7) and a MAP correction (negated gate, threshold about +0.7), and keep both off for drug atoms. That also lets the correction reach glucose_lab and map_cuff, which the evaluation scores too.

The bin centres are in real units. Glucose_finger bins are about 9–22 mg/dL wide, and map_arterial bins about 3–7 mmHg.

Before choosing which hidden units to reuse, I need the cost of freezing each one at its mean. Because the model sits at a likelihood optimum, the first-order cost is about zero. The second-order cost is about ½·Var(activation)·w_uᵀ(diag p − ppᵀ)w_u, which needs only the predicted probabilities, not targets.

I have helper functions from before but don't remember their signatures. So this run prints the source of my head-prediction helpers and the column layout of the reading-level tables, so I compute this cost with the right arrays. If the helpers reproduce the logits from PREall, I can compute the cost for all 192 units in the next run.

```python
import inspect
for fn_name in ['gelu', 'head_pred', 'head_means', 'predict3']:
    try:
        src = inspect.getsource(globals()[fn_name])
        print(f"---- {fn_name} ----\n{src[:1500]}")
    except Exception as ex_:
        print(fn_name, "source unavailable:", ex_)
print("R99 cols:", list(R99.columns))
print(R99.head(3).to_string())
print("M4 cols:", list(M4.columns))
for nm_ in ['TR99','CENm','EI','EN','G4','err4','pg','isG','isM','codeall','kidx','lb_','pmall','m_','mA2']:
    vv = globals()[nm_]
    print(nm_, getattr(vv,'shape',None), np.round(np.asarray(vv).ravel()[:6].astype(float),3))
```