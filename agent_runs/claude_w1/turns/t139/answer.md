Run 129 showed three things:
- The dosing rules in the record are step functions of the reading taken that hour. Insulin switches on above about 160 and has rate levels near 180/220/260; norepinephrine thresholds are 65/75.
- The value head has 17 classes, with a (code × last-bin) logit table (`embed_transition.value_bin`, 221×17) and a 192→192→17 MLP. A per-last-bin offset can therefore be written straight into that table, which helps cancel the part of the dose that the dosing rule predicts.
- The 35,007-row block is not the one-step factual residual: its residual sd is 3.2, not ~14. I will set it aside rather than guess what it is.

My plan: tilt the glucose and MAP logits in proportion to the decoded lagged dose minus the rule-expected dose. The decoders will be trained on locally simulated states with scaled doses, so that they respond to the dose itself. I'll then calibrate the gain with dose_edit runs on setA against my truth estimates.

Function sources aren't retrievable, so first I will inspect the code objects of my generator and replica functions to recall their behaviour. I will also read the value-bin note and centres, which I need to turn a mean shift into logits. Success means I can tell how `gen_samples` perturbs doses and how `bundles_of_stay`'s `dscale` works, and I know the bin centres and how the mean is computed.

```python
def cinfo(f):
    c = f.__code__
    return dict(args=c.co_varnames[:c.co_argcount], names=[n for n in c.co_names][:40], consts=[str(k)[:30] for k in c.co_consts if not hasattr(k,'co_code')][:25], defaults=f.__defaults__)
for fn_ in [gen_samples, bundles_of_stay, run_stay, wc_vec, head_value2, true_shift, truth_rows]:
    print(fn_.__name__, cinfo(fn_)); print()
print(str(VOCAB['value_bin_note'])[:1500])
vbc_ = VOCAB['value_bin_centres']
print(type(vbc_), list(vbc_.keys())[:6] if isinstance(vbc_, dict) else len(vbc_))
for c_ in ['glucose_finger','map_arterial']:
    print(c_, np.round(np.asarray(vbc_[c_]),1).tolist() if isinstance(vbc_, dict) and c_ in vbc_ else '')
```