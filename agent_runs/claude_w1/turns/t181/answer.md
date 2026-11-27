The notebook's earlier sections are back. Truth on setA from the record-based impulse responses: insulin x2 −14.17, x1/2 +7.09, NE x2 +4.82, NE x1/2 −2.41. The original model gives −1.45 / +0.82 / +0.15 / −0.41, and its factual NLL on setA is 1.51903.

The current plan is a head-level correction of the form c·(decoded dose − its mean given code and last reading bin). That centring cancels the policy part the model has already absorbed, so in theory factual predictions stay unbiased while the edited dose shifts the prediction. With the model's own persistence of about 0.78 per hour, a one-step insulin correction of about −3 per u/h should give roughly the true time course.

The NE side is the bottleneck. Decoding error (sd about 0.19) is large relative to the NE innovation (sd 0.079), so the needed gain of about 4–5 would use almost the whole factual budget. That calls for a better decoder: minimum factual variance subject to unit dose response, plus a code gate.

Before building it I need to see the signatures of the kernel helpers I will reuse. I will print only short summaries to keep the output small.

```python
import inspect
for fn in ['run_stay','gen_samples','head_mean','head_P','bundles_of_stay','core_step','run_rows','run_rows_wc','fit_dec','mvdr','lastbin_of','eval_head','gen_many','nu_of']:
    f_ = globals().get(fn)
    if f_ is None: print(fn,'MISSING'); continue
    try: print(fn, str(inspect.signature(f_))[:150])
    except Exception as ex: print(fn,'?',type(f_))
for nm in ['W1','b1','W2','b2','TR','CENT','H100','DEC','samp_pert','setA','SAMP','F2df']:
    v=globals().get(nm)
    if v is None: print(nm,'MISSING'); continue
    s=getattr(v,'shape',None)
    print(nm, type(v).__name__, s if s is not None else (len(v) if hasattr(v,'__len__') else ''))
print([k for k in DEC.keys()][:20] if isinstance(globals().get('DEC'),dict) else '')
print([n for n in NAMES if 'attr_heads.value' in n or 'transition' in n][:20])
```