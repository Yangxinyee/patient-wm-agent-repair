The core has the structure I expected. A pre-norm prelude layer (attention, then a 768-wide FFN) runs over the 7 tokens. Then come two recurrent blocks and a coda layer. None of the attention projections have biases.

The dose enters the state only through the prelude: the slot tokens attend to the u_k token and to the z_in(z_k) token. So a clean dose channel must be written through the prelude's value projection, in the u_k or z_in(z_k) token's contribution to the slot outputs.

How well that works depends on how much attention the slots pay to those tokens and how stable it is. So next I'll read the model config (number of heads) and test my rebuilt drug-bundle encoder on a few drug bundles. It needs to reproduce the posterior z that the tool gives in the state files; that is what the earlier 0.056 error check was about.

If the encoder rebuild matches, I'll reconstruct the prelude attention at drug bundles and measure the attention weights on the u token.

```python
print(VOCAB['config'])
print('S2 keys', list(S2.keys()) if hasattr(S2,'keys') else type(S2))
print('jsR[:5]', jsR[:5], 'errs[:5]', np.round(errs[:5],3))
import inspect
for fn in [drug_atoms, setenc, zmean, ctx_from, enc_a, meas_atoms, e_empty, err_route4]:
    try: print(fn.__name__, inspect.signature(fn), fn.__code__.co_varnames[:12], [c for c in fn.__code__.co_names][:15])
    except Exception as e: print(fn, e)
```