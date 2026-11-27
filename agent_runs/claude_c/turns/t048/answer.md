Double counting is a real risk. A lag-1 bias of −1.98 × expected tier rate, applied to the finger readings after insulin (64% of finger readings), would cost about 0.0086 nats per reading. That alone breaks the 0.005 limit, before any lagged bias.

The original model's residuals at those readings fall with the dose innovation, slope −1.37 per U/h. So capturing the true innovation effect would slightly improve the factual fit. The goal is a response to the dose's deviation from the tier policy, not to the raw dose.

To choose between "amplify the existing pathway" and "build a compensated edit", I need to know how the model's current insulin response is spread over hours 13–24. I will compare the per-hour shifts of the original and edited model in pass 4 with my per-hour truth.

- If the hour-13 (lag-1) shift is near zero and the response builds up later, the model's dose effect runs through slow state memory.
- If it is front-loaded, the dose acts mainly on the next prediction.

```python
import inspect
de_o = pd.read_csv('/work/fwd_4/dose_edit_original.csv'); de_e = pd.read_csv('/work/fwd_4/dose_edit_edited.csv')
print(de_o.columns.tolist()); print(de_o.head(3))
for fn_ in ['true_effect_table','pooled_t','first_dose_hour','stays_in']:
    if fn_ in globals() and callable(globals()[fn_]):
        try: print(inspect.getsource(globals()[fn_])[:1500])
        except Exception as e_: print(fn_, 'no source', e_)
print(type(tt), getattr(tt,'shape',None), list(tt.columns) if hasattr(tt,'columns') else '')
```