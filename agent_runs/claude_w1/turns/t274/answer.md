`gen_samples` reads its weights from `Wo` (float64 numpy arrays), so I can simulate candidate edits locally by editing `Wo` and then restoring it. Scaling the prelude output projection by 1.5 moved the head inputs by only 0.011 on average, which shows how strongly the LayerNorms and the state refresh damp upstream changes.

The plan is to inject the dose through prelude head 2, which attends almost fully to the u token. That needs the u-token representation and the decoders `w_ins` / `w_ne` on `norm1(u)`. Next I'll check the signatures of the encoder helpers and the structure of a drug bundle. Then I can confirm that `w_ins·norm1(u)` reproduces the recorded insulin dose.

```python
for fn in ['encset','avec','bundles_of_stay','zpost_of','ctx_of','atoms_of_stay','init_ctx','mha','ffn','LNn']:
    try: print(fn, inspect.signature(g[fn]))
    except Exception as e: print(fn, 'ERR', str(e)[:50])
print("w_ins", getattr(w_ins,'shape',type(w_ins)), "w_ne", getattr(w_ne,'shape',type(w_ne)), "w_", getattr(w_,'shape',type(w_)))
B0_ = bundles_of_stay(G3[s0_])
print(type(B0_).__name__, len(B0_))
b_ex = B0_[3] if isinstance(B0_, list) else None
print(type(b_ex).__name__, (list(b_ex.keys()) if isinstance(b_ex, dict) else (len(b_ex) if b_ex is not None else None)))
```