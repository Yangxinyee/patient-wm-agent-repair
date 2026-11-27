The kernel-only helpers (`make_W_chan`, `make_core_chan`, `eval_wc`, `eval_cfg`) have no source file. The kernels are now clear:

- `Kins` and `Knor` are the structural impulse responses (insulin on glucose, norepinephrine on MAP).
- `G1`/`Gf` are the one-step increments assuming the model propagates shifts fully; `G85` assumes 85% propagation; `Gp` is another variant.

`corechan.npz` holds two 192-vectors (`vI`, `vN`), two 48-vectors and a scale, which suggests a channel through one attention head (48 = 192/4). To reuse it safely I need to know which tensors these builders edit and what they take as arguments. I'll inspect their signatures and the names their code refers to, without printing raw listings.

```python
import inspect
for nm in ['make_core_chan','make_W_chan','eval_wc','eval_cfg','hd_shift','pair_diffs','run_rollout_h2','run_rollout_h']:
    f = globals()[nm]; c = f.__code__
    try: sig = str(inspect.signature(f))
    except Exception as e: sig = '?'
    strs = [s for s in c.co_consts if isinstance(s,str)][:12]
    print(nm, sig[:200]); print('   names:', ' '.join(c.co_names[:40])[:400]); print('   strs:', strs)
```