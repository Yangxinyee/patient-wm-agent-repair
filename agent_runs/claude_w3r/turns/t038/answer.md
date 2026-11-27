This result is decisive. The state encodes the band mean strongly (R² 0.90, persisting for hours), but the dose deviation only weakly, and only right at the insulin bundle (R² 0.24). It is gone by the next bundle (R² 0.02). The model essentially forgets the insulin dose immediately. A head-only edit therefore cannot produce a multi-hour insulin effect; the core must be edited to carry dose information forward. Designing such an edit safely needs a local re-implementation of the model, validated against these states, so I can test edits on CPU before spending forward passes.

As the first step of re-implementing, I will infer how records become bundles and atoms. I will compare one stay's events with the bundle hours and atom counts from the states file. That should settle, for example, whether infusion records produce separate END atoms an hour later.

```python
with open('/work/notes.md','a') as f:
    f.write("""- FWD2 states (orig, train selS 100 stays w/ both drugs+art line; /work/fwd_2). Probes (ridge, grouped CV):
  insulin rate at insulin bundle R2 ~0.65-0.69; band-mean c R2 0.90 (persists: .89,.86,.82 at later bundles);
  DEVIATION r-c: R2 0.24 at the insulin bundle, 0.02 one bundle later, ~0 after -> model FORGETS dose deviation immediately.
  => head-only edit can't give multi-hour effect; need core edit carrying dose memory. Plan: local re-implementation of model
  (validate vs states), save code in /work/impl.py.
- True-effect structure guess (insulin): AR(1) glucose rho~0.75 + direct effects b1=-2.0,b2=-0.95,b3=-0.26 reproduces IRF.
""")
s0 = selS[0]
ev0 = tr[tr.stay_id==s0].sort_values('hour')
print(ev0[['hour','event','value','rate','dose','end_hour']].head(40).to_string())
b0 = B[B.stay_id==s0][['hour','na','hi','hn']]
print(b0.head(30).to_string())
print(b0.tail(5).to_string())
```