The z posterior is not a clean "dose departure" signal.

At the 2,000 insulin bundles:
- z predicts the recorded rate with R² 0.73 and the policy-expected rate with R² 0.74, but the departure only with R² 0.32.
- `out` predicts the expected rate with R² 0.93, the recorded rate with 0.80 and the departure with 0.46.
- So the state knows the glucose band almost exactly, but the dose itself only coarsely (error about 0.4 U/h).

NE is similar: `out` predicts the rate with R² 0.80 and the departure with 0.26.

Conclusion: amplifying only the z path won't isolate the departure. A simple linear "departure" term added to z_post would also fire at every non-drug bundle, because a linear layer cannot gate on drug presence.

The design I now favour is a direct, gated readout in the value head:
- it is switched on only for glucose codes (insulin) or MAP codes (NE), using the code embedding;
- it reads a linear "departure" estimate from the head input, because the state encodes the glucose band well;
- it shifts the predicted distribution by a tilt across bins.

Because the departure estimate tracks the model's own sampled glucose in a rollout, it also cancels the counter-effect the model has folded into its glucose dynamics.

Before building it I need to recover what my helper functions compute: their arguments, which activation they use (GELU or ReLU), and which tensors they read. I also need three facts about the weights:
- whether `c_out` is close to the identity;
- what the 8 columns of `last_numeric` are;
- how the value-head hidden units are used.

```python
for fn in ['mlp','lin','ln','atom_vec','atom_vec2','encode','ctx_from','ctx_v','zpred','zsc','fbin','stay_atoms','T','band','deconv','true_shift_rows','build_grid']:
    f_ = globals()[fn]; co = f_.__code__
    print(fn, co.co_varnames[:co.co_argcount], '| names:', co.co_names[:25], '| consts:', [c_ for c_ in co.co_consts if isinstance(c_,(str,float,int))][:12])
Wc = load('core.c_out.weight'); I_ = torch.eye(192)
print('c_out: |W-I|/|I|', float((Wc-I_).norm()/I_.norm()), 'diag mean', float(Wc.diag().mean()), 'fro', float(Wc.norm()))
print('last_numeric col norms', load('last_numeric.weight').norm(dim=0).numpy().round(2))
print('embed.numeric col norms', load('embed.numeric.weight').norm(dim=0).numpy().round(2))
print(VOCAB['numeric_note'] if 'numeric_note' in VOCAB else '', VOCAB['value_bin_note'])
print({k2: (v2 if not isinstance(v2,(list,dict)) else (list(v2)[:20] if isinstance(v2,dict) else v2[:20])) for k2,v2 in VOCAB['numeric'].items()} if isinstance(VOCAB['numeric'],dict) else VOCAB['numeric'])
```