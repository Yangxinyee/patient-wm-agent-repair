The last output brought back my notebook, the architecture description and the kernel's function list. The run-66 errors were all in dimensions below 32. They were therefore errors in the z-posterior mean: I have re-implemented the embedding, the two set encoders and `z_post`, and checked them against the tool's states. There seems to be no function yet for the full core step (prelude, recurrent block, coda) or the value head.

My plan is still the one in the notebook. The model only keeps about one hour of the insulin dose signal, so I want to give it a multi-hour dose memory inside the core and have the value head read it. To design and test that edit in the sandbox without spending forward passes, I need an exact copy of the core step.

Next I will list the core and head tensors and reread the source of my existing re-implementation functions, so I can build the core step on them. If `core.coda` / `core.c_out` turn out to be a simple linear plus norm, a slowly decaying memory channel through `h` is feasible. If the coda renormalises `h` heavily, I will need a different route, for example through `E`/`b_in` dimensions with slow `a_bar`.

```python
import inspect, json
core=[(n,SHAPES[n]) for n in NAMES if n.startswith('core.')]
print(len(core))
for n,s in core: print(n,s)
print('--- heads/other')
for n in NAMES:
    if not n.startswith('core.') and not n.startswith('embed.') and not n.startswith('encode'):
        print(n,SHAPES[n])
for fn in ['atom_vec','set_encode','build_bundles','z_pred_k','LN','mlp','lin']:
    try:
        src=inspect.getsource(globals()[fn]); print('#####',fn,len(src)); print(src[:1500])
    except Exception as ex: print(fn,'ERR',ex)
```